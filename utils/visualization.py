import torch, torchvision
from typing import Dict, List
from PIL import Image

def convert_segmentation_to_img(tensor: torch.Tensor,
                                color_lookup: Dict[int, List[int]] = {0: [0, 0, 0],
                                                                      1: [255, 0, 0],
                                                                      2: [0, 255, 0],
                                                                      3: [0, 0, 255],
                                                                      4: [255, 255, 0],
                                                                      5: [128, 128, 128],
                                                                      6: [128, 128, 0],
                                                                      7: [0, 128, 128]}) -> torch.Tensor:
    """
    Args:
        tensor: Segmentation tensor with classes at every pixel
        color_lookup: Dict to map every class to a rgb color

    Returns:
        out_tensor: Segmentation ensor as rgb image
    """
    out_tensor = torch.zeros(tensor.shape, device=tensor.device).repeat(1, 3, 1, 1)
    for unique_ind in torch.unique(tensor):
        rgb = color_lookup[int(unique_ind)]
        for col_idx, col_val in enumerate(rgb):
            for batch_idx, tensor_batch in enumerate(tensor):
                out_tensor[batch_idx, col_idx] = (tensor_batch == unique_ind).squeeze(0) * col_val/255

    return out_tensor

def img_path_to_tensor(img_path: str) -> torch.Tensor:
    """
    Args:
        img_path: Path to a rgb image

    Returns:
        tensor: Image as a NCHW tensor
    """
    return torch.stack([torchvision.transforms.functional.to_tensor(Image.open(path))
                        for path in img_path])

def blend_images(img: torch.Tensor, seg: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Args:
        img: NCHW image
        seg: NCHW segmentation
        alpha: Value to determine blending

    Returns:
        blended_tensor: blended img and seg images
    """
    blended_tensors = []
    for img_sample, seg_sample in zip(img, seg):
        img_pil = torchvision.transforms.ToPILImage()(img_sample)
        seg_pil = torchvision.transforms.ToPILImage()(seg_sample)
        # blended_pil = Image.blend(img_pil, seg_pil, alpha=alpha)
        blended_pil = Image.blend(seg_pil, img_pil, alpha=alpha)
        blended_tensor = torchvision.transforms.ToTensor()(blended_pil)
        blended_tensors.append(blended_tensor)
    blended_tensors = torch.stack(blended_tensors)
    return blended_tensors