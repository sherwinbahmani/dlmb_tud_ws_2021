from torchvision.models import segmentation
import torch

def deeplabv3_resnet50(num_classes: int,
                       pretrained: bool = True) -> torch.nn.Module:
    """
    Loads pretrained model from torchvision and appends custom head with num_classes

    Args:
        pretrained: If True, model is pretrained on coco
        num_classes: Number of classes of chosen dataset
    """
    # Get model with pretrained coco weights
    model = segmentation.deeplabv3_resnet50(pretrained=pretrained)

    # Reinitialize classifier after loading weights with num_classes
    model.classifier[-1] = classification_head(model.classifier[1].out_channels,
                                               num_classes)

    model.aux_classifier = None
    return model

def fcn_resnet50(num_classes: int,
                 pretrained: bool = True) -> torch.nn.Module:

    # Get model with pretrained coco weights
    model = segmentation.fcn_resnet50(pretrained=pretrained)

    # Reinitialize classifier after loading weights with num_classes
    model.classifier[-1] = classification_head(model.classifier[0].out_channels,
                                               num_classes)

    model.aux_classifier = None
    return model

def classification_head(in_channels: int,
                        out_channels: int) -> torch.nn.Module:
    """
    Creates a custom classification head
    Args:
        in_channels: Num channels of previous layer
        out_channels: Num classes
    Returns:
        class_conv: Convolution to reduce num channels to num classes
    """
    head = torch.nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

    torch.nn.init.normal_(head.weight.data, mean=0, std=0.01)
    torch.nn.init.constant_(head.bias.data, 0.0)

    return head