import os, pathlib
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm

import models
from loss.semantic_seg import CrossEntropyLoss
from datasets.dataset import HAM10000Dataset
from optimizer.schedulers import *
from utils.metrics import *
from utils import transforms


def main(opts):

    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Add tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(opts.runs_root, time_stamp))

    # Add checkpoints results directory
    pathlib.Path(opts.checkpoints_root).mkdir(parents=True, exist_ok=True)
    checkpoints_path = os.path.join(opts.checkpoints_root,
                                    time_stamp + ".pth")

    # Setup Metrics
    iou_meter = runningScore(opts.num_classes)

    # Run on GPU if available else on CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup dataset
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(opts.crop_size),
                                           transforms.RandomHFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize()])
    train_dataset = HAM10000Dataset(root=opts.dataset_root,
                                    target_type='train',
                                    transforms=train_transforms)
    class_names = list(train_dataset.labels_lookup.keys())
    train_loader = DataLoader(train_dataset,
                              batch_size=opts.batch_size,
                              shuffle=True,
                              num_workers=opts.num_workers)
    val_transforms = transforms.Compose([transforms.CenterCrop(opts.crop_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize()])
    val_dataset = HAM10000Dataset(root=opts.dataset_root,
                                  target_type='val',
                                  transforms=val_transforms)
    val_loader = DataLoader(val_dataset,
                            batch_size=opts.batch_size,
                            shuffle=False,
                            num_workers=opts.num_workers)
    # Setup model
    model = models.__dict__[opts.model_name](pretrained=opts.pretrained,
                                             num_classes=opts.num_classes)

    # Allow multiple gpus
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Setup lr scheduler, optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opts.base_lr,
                                momentum=opts.momentum,
                                weight_decay=opts.weight_decay)

    scheduler = PolyLR(optimizer=optimizer,
                       max_iter=len(train_loader)*opts.num_epochs+1,
                       gamma=opts.lr_gamma)

    criterion = CrossEntropyLoss().to(device)

    # Training
    mean_iou_best = 0
    for epoch in tqdm(range(opts.num_epochs)):
        loss_train_epoch, loss_val_epoch = 0, 0
        for train_idx, train_sample in enumerate(train_loader):

            # Put img and gt on GPU if available
            img_train, gt_train = train_sample['img'].to(device), train_sample['gt'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass, backward pass and optimization
            out_train = model(img_train)['out']
            loss_train = criterion(out_train, gt_train)
            loss_train_epoch += loss_train
            loss_train.backward()
            optimizer.step()

            # Determine current lr
            scheduler.step()
            writer_idx = train_idx + len(train_loader)*epoch

            # Add current loss to tensorboard
            writer.add_scalar("training_loss_step",
                              loss_train,
                              writer_idx)
            writer.add_scalar("learning_rate",
                              optimizer.param_groups[0]['lr'],
                              writer_idx)

        # Add mean epoch loss to tensorboard
        writer.add_scalar("training_loss_epoch",
                          loss_train_epoch/len(train_loader),
                          epoch)

        # Validation
        if epoch >= opts.validation_start and epoch % opts.validation_step == 0:
            model.eval()
            with torch.no_grad():
                for val_idx, val_sample in enumerate(val_loader):

                    # Put img and gt on GPU if available
                    img_val, gt_val = val_sample['img'].to(device), val_sample['gt'].to(device)

                    # Forward pass and loss calculation
                    out_val = model(img_val)['out']
                    loss_val = criterion(out_val, gt_val)
                    loss_val_epoch += loss_val

                    # Update iou meter
                    iou_meter.update(gt_val.cpu().numpy(), torch.argmax(out_val, dim=1).cpu().numpy())

            model.train()

            # Update metrics and add to tensorboard
            score, class_iou, cm, iu = iou_meter.get_scores()

            # Compute tensorboard confusion matrix figure and class iou table
            cm_figure = plot_confusion_matrix(cm, class_names)
            ciou_figure = plot_class_table(class_iou, class_names, column_head='IoU per class')
            writer.add_figure("Confusion Matrix", cm_figure, epoch)
            writer.add_figure("Class IoU Table", ciou_figure, epoch)

            for k, v in score.items():
                writer.add_scalar(f"val_metrics/{k}", v, epoch)

            # Reset metrics
            iou_meter.reset()

            # Add loss to tensorboard
            writer.add_scalar("validation_loss", loss_val_epoch/len(val_loader), epoch)

            # Save model
            mean_iou_epoch = score['Mean IoU :']
            if mean_iou_epoch > mean_iou_best:
                if os.path.exists(checkpoints_path):
                    os.remove(checkpoints_path)
                torch.save(model.state_dict(), checkpoints_path)
                mean_iou_best = mean_iou_epoch


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
        default=os.path.join(os.getcwd(), "checkpoints", "runs")
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=os.path.join(os.getcwd(), "runs")
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2
    )
    parser.add_argument(
        "--validation-start",
        type=int,
        default=0
    )
    parser.add_argument(
        "--validation-step",
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
        "--pretrained",
        action="store_true",
        help="Calling this means pretrained on coco, not calling means only on ImageNet"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=1e-3
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=406
    )

    clargs = parser.parse_args()
    main(clargs)