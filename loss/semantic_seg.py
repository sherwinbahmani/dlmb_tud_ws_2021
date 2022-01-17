import torch

class CrossEntropyLoss(torch.nn.Module):

    def __init__(self):

        super(CrossEntropyLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self,
                output: torch.Tensor,
                gt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: Probabilities for every pixel
            gt: Labeled image at full resolution

        Returns:
            loss: Cross entropy loss
        """
        return self.criterion(output, gt.long().squeeze(1))