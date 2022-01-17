import numpy as np
import pandas as pd
from typing import List, Dict
import os, pathlib, shutil
from argparse import ArgumentParser

def split_raw_dataset(opts,
                      zip_files: List[str] = ["HAM10000_images_part_1",
                                              "HAM10000_images_part_2",
                                              "HAM10000_segmentations_lesion_tschandl"],
                      label_file: str = "HAM10000_metadata.csv",
                      split_ratio: Dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}):
    """
    Splits the initial dataset downloaded into train, test and val datasets

    Args:
        opts: Arguments inclunding path to raw dataset
        zip_files: Name of zip files of images and segmentation masks
        label_file: csv file with type of skin disease
        split_ratio: Defines how to split the dataset
    """
    # Get extracted zip file paths
    zip_files_path = [os.path.join(opts.dataset_raw, zip_file) for zip_file in zip_files]
    for zip_file_path in zip_files_path:
        assert os.path.exists(zip_file_path), f"Path {zip_file_path} does not exist, please extract the zip file"

    # Read labels
    labels_df = pd.read_csv(os.path.join(opts.dataset_raw, label_file), sep=',')
    labels_lookup = labels_df.values[:, 1:3]
    # Randomly shuffle before reading names
    np.random.shuffle(labels_lookup)
    img_names = labels_lookup[:, 0]

    # Split into train, val and test
    img_names_split = {}
    start_idx, last_idx = 0, 0
    for k, v in split_ratio.items():
        num_images = int(np.ceil(v*len(img_names)))
        last_idx += num_images
        img_names_split[k] = img_names[start_idx:last_idx]
        start_idx += num_images

    # Copy images and groundtruth paths
    out_path = opts.dataset_raw.replace("raw", "split")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    for split_k, split_v in img_names_split.items():
        out_path_split = os.path.join(out_path, split_k)
        out_path_images = os.path.join(out_path_split, "img")
        out_path_gts = os.path.join(out_path_split, "gt")
        pathlib.Path(out_path_images).mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_path_gts).mkdir(parents=True, exist_ok=True)
        out_labels_list = []
        for img_name in split_v:
            # Image
            img_file_exists = False
            for zip_file in zip_files[:-1]:
                img_path = os.path.join(opts.dataset_raw, zip_file, img_name + ".jpg")
                if os.path.isfile(img_path):
                    img_file_exists = True
                    break
            assert img_file_exists, f"Image {img_name} does not exist in any of the two image directories"
            out_path_image = os.path.join(out_path_images, img_name + ".jpg")
            shutil.copy2(img_path, out_path_image)
            # Groundtruth mask
            gt_name = img_name + "_segmentation.png"
            gt_path = os.path.join(opts.dataset_raw, zip_files[-1], gt_name)
            out_path_gt = os.path.join(out_path_gts, gt_name)
            shutil.copy2(gt_path, out_path_gt)
            # Set label
            label_idx = np.where(labels_lookup[:, 0] == img_name)[0][0]
            out_labels_list.append([img_name, labels_lookup[label_idx, 1]])

        # Save labels as csv
        out_labels_arr = np.array(out_labels_list)
        out_labels_path = os.path.join(out_path_split, "labels.csv")
        np.savetxt(out_labels_path, out_labels_arr, delimiter=",", fmt='%s')

if __name__ == "__main__":
    parser = ArgumentParser()
    # Path to directory ../dataset/raw
    parser.add_argument(
        "--dataset-raw",
        type=str,
        default=os.path.join(pathlib.Path(os.getcwd()).parents[1], "dataset", "raw")
    )
    clargs = parser.parse_args()
    split_raw_dataset(clargs)