import pandas as pd
import os, torch
from typing import Callable, Dict, Union
from PIL import Image
from torch.utils.data import Dataset

class HAM10000Dataset(Dataset):

    def __init__(self,
                 root: str,
                 target_type: str,
                 transforms: Callable):

        # Set up paths to images and groundtruths
        self.target_type = target_type
        self.root = os.path.join(root, target_type)
        labels = pd.read_csv(os.path.join(self.root, "labels.csv"), sep=',').values
        names = labels[:,0]
        self.labels_list = labels[:,1]
        self.img_list, self.gt_list = [], []
        for name in names:
            self.img_list.append(os.path.join(self.root, "img", name + ".jpg"))
            self.gt_list.append(os.path.join(self.root, "gt", name + "_segmentation.png"))

        # Lookup to map label strings to integers (0 is background)
        self.labels_lookup = {"background": 0, "akiec": 1, "bcc": 2, "bkl": 3, "df": 4, "nv": 5, "vasc": 6, "mel": 7}
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self,
                    idx: int) -> Dict[str, Union[torch.Tensor, str]]:

        # Read image, groundtruth segmentation and convert label to index
        img = Image.open(self.img_list[idx])
        gt = Image.open(self.gt_list[idx])
        label = self.labels_lookup[self.labels_list[idx]]

        # Transform images to tensors and apply data augmentation
        img, gt = self.transforms(img, gt)

        # Multiply groundtruth with label
        gt //= 255
        gt *= label

        return {'img': img, 'gt': gt, 'img_path': self.img_list[idx]}