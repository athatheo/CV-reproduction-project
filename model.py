import torch
from torch.nn import Module
from HRNet.model import HRNet
from CSP_Head.csp_head import Detection
from FSH.model import FSH

class F2DNet(Module):
    def __init__(self, hrnet, bbox_head, refine_head):
        super(F2DNet, self).__init__()
        self.hrnet = hrnet
        self.focal_detection_network = bbox_head
        self.fast_suppresion_head = refine_head

    def forward(self, x):
        x = self.hrnet_backbone(x)
        x = self.hrnet_fpn(x)
        detections = self.focal_detection_network(x)
        # Maybe a refine_roi_extractor should be added here
        x = self.fast_suppresion_head(x+detections)
        return x


