import os, torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from tqdm import tqdm
from datetime import datetime

import models
from datasets.dataset import HAM10000Dataset
from utils.metrics import *
from utils import transforms
from utils import visualization

def main(opts):

    # Add tensorboard writer
    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    writer = SummaryWriter(log_dir=os.path.join(opts.runs_root, time_stamp + "_inference"))

    # Setup Metrics
    iou_meter = runningScore(opts.num_classes)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup test dataset
    test_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize()])
    test_dataset = HAM10000Dataset(root=opts.dataset_root,
                                   target_type='test',
                                   transforms=test_transforms)
    class_names = list(test_dataset.labels_lookup.keys())

    test_loader = DataLoader(test_dataset,
                             batch_size=opts.batch_size,
                             shuffle=False,
                             num_workers=opts.num_workers)

    # Setup model
    model = models.__dict__[opts.model_name](pretrained=True,
                                             num_classes=opts.num_classes)

    # Allow multiple gpus
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)

    # Pick newest checkpoints
    checkpoint = os.path.join(opts.checkpoints_root, opts.checkpoint_name)
    if os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError(f"Checkpoints file {checkpoint} does not exist")

    model.eval()
    with torch.no_grad():
        for test_idx, test_sample in enumerate(tqdm(test_loader)):

            # Send inputs to device
            img_test, gt_test = test_sample['img'].to(device), test_sample['gt'].to(device)

            # Forward pass
            out_test = model(img_test)['out']
            pred_test = torch.argmax(out_test, dim=1, keepdim=True)

            # Argmax for semantic prediction
            iou_meter.update(gt_test.cpu().numpy(), pred_test.cpu().numpy())

            # Visualize image, gt and prediciton
            img_test_orig = visualization.img_path_to_tensor(test_sample['img_path'])
            gt_test = visualization.convert_segmentation_to_img(gt_test)
            pred_test = visualization.convert_segmentation_to_img(pred_test)
            gt_test_blend = visualization.blend_images(img_test_orig, gt_test)
            pred_test_blend = visualization.blend_images(img_test_orig, pred_test)
            writer.add_images("Image", img_test_orig, test_idx)
            writer.add_images("Groundtruth original", gt_test, test_idx)
            writer.add_images("Prediction original", pred_test, test_idx)
            writer.add_images("Groundtruth", gt_test_blend, test_idx)
            writer.add_images("Prediction", pred_test_blend, test_idx)

        # Add predictions to tensorboard
        score, class_iou, cm, iu = iou_meter.get_scores()
        for k, v in score.items():
            writer.add_scalar(f"test_metrics/{k}", v, 0)

        # Compute tensorboard confusion matrix figure and class iou table
        cm_figure = plot_confusion_matrix(cm, class_names)
        ciou_figure = plot_class_table(class_iou, class_names, column_head='IoU per class')
        writer.add_figure("Confusion Matrix", cm_figure, 0)
        writer.add_figure("Class IoU Table", ciou_figure, 0)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.join(os.path.dirname(os.getcwd()), "dataset", "split")
    )
    parser.add_argument(
        "--checkpoints-root",
        type=str,
        default=os.path.join(os.getcwd(), "checkpoints", "eval")
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=os.path.join(os.getcwd(), "runs")
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=8
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="fcn_resnet50",
        choices=['deeplabv3_resnet50', 'fcn_resnet50']
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    clargs = parser.parse_args()
    main(clargs)