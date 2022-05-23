import torch
from torch.nn import Module


class F2DNet(Module):
    def __init__(self, backbone, neck, bbox_head, refine_head):
        super(F2DNet, self).__init__()
        self.hrnet_backbone = backbone
        self.hrnet_fpn = neck
        self.focal_detection_network = bbox_head
        self.fast_suppresion_head = refine_head

    def forward(self, x):
        x = self.hrnet_backbone(x)
        x = self.hrnet_fpn(x)
        detections = self.focal_detection_network(x)
        # Maybe a refine_roi_extractor should be added here
        x = self.fast_suppresion_head(x+detections)
        return x

class HRNet_backbone(Module):
    def __init__(self, backbone):
        super(HRNet_backbone, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return x

# Create class for every F2DNet model
class HRNet_fpn(Module):
    def __init__(self, fpn):
        super(HRNet_fpn, self).__init__()
        self.fpn = fpn

    def forward(self, x):
        x = self.fpn(x)
        return x


class Focus_detection_network(Module):
    def __init__(self, detection_network):
        super(Focus_detection_network, self).__init__()
        self.detection_network = detection_network

    def forward(self, x):
        x = self.detection_network(x)
        return x


class Fast_suppression_head(Module):
    def __init__(self, fast_suppression_head):
        super(Fast_suppression_head, self).__init__()
        self.fast_suppression_head = fast_suppression_head

    def forward(self, x):
        x = self.fast_suppression_head(x)
        return x
