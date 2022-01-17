import torch, random
import torchvision.transforms.functional as F
import torchvision.transforms as tf
import numpy as np
from PIL import Image
from typing import Tuple, List, Callable

class Compose:

    def __init__(self,
                 transforms: List[Callable]):

        self.transforms = transforms

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:

        for transform in self.transforms:
            img, gt = transform(img, gt)

        return img, gt

class ToTensor:

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:

        img = F.to_tensor(np.array(img))
        gt = torch.from_numpy(np.array(gt)).unsqueeze(0)

        return img, gt

class Resize:

    def __init__(self,
                 resize: Tuple[int]):

        self.img_resize = tf.Resize(size=resize,
                                    interpolation=Image.BILINEAR)
        self.gt_resize = tf.Resize(size=resize,
                                   interpolation=Image.NEAREST)

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[Image.Image, Image.Image]:

        img = self.img_resize(img)
        gt = self.gt_resize(gt)

        return img, gt


class Normalize:

    def __init__(self,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):

        self.norm = tf.Normalize(mean=mean,
                                 std=std)

    def __call__(self,
                 img: torch.Tensor,
                 gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        img = self.norm(img)

        return img, gt


class RandomHFlip:

    def __init__(self,
                 percentage: float = 0.5):

        self.percentage = percentage

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[Image.Image, Image.Image]:

        if random.random() < self.percentage:
            img = F.hflip(img)
            gt = F.hflip(gt)

        return img, gt


class RandomResizedCrop:

    def __init__(self,
                 crop_size: int):

        self.crop = tf.RandomResizedCrop(size=crop_size)

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[Image.Image, Image.Image]:

        i, j, h, w = self.crop.get_params(img=img,
                                          scale=self.crop.scale,
                                          ratio=self.crop.ratio)
        img = F.resized_crop(img, i, j, h, w, self.crop.size, Image.BILINEAR)
        gt = F.resized_crop(gt, i, j, h, w, self.crop.size, Image.NEAREST)

        return img, gt

class CenterCrop:

    def __init__(self,
                 crop_size: int):

        self.crop = tf.CenterCrop(size=crop_size)

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[Image.Image, Image.Image]:

        img = self.crop(img)
        gt = self.crop(gt)

        return img, gt