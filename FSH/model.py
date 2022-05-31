import torch
from torch.nn import Module

class FSH(Module):
    def __init__(self, fast_suppression_head):
        super(FSH, self).__init__()
        self.fast_suppression_head = fast_suppression_head

    def forward(self, x):
        x = self.fast_suppression_head(x)
        return x