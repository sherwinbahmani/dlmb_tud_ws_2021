'''
Source: https://github.com/meetshah1995/pytorch-semseg
'''

from torch.optim.lr_scheduler import _LRScheduler
import torch
from typing import List

class PolyLR(_LRScheduler):
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 max_iter: int,
                 decay_iter: int = 1,
                 gamma: float = 0.9,
                 last_epoch: int = -1):

        self.max_iter = max_iter
        self.decay_iter = decay_iter
        self.gamma = gamma
        self.factor: float
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Returns:
            lr: Current learning rate based on iteration
        """
        assert self.last_epoch < self.max_iter\
            , f"Last epoch is {self.last_epoch} but needs to be smaller than max iter {self.max_iter}"
        self.factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
        return [base_lr * self.factor for base_lr in self.base_lrs]