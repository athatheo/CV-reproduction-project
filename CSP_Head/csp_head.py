import torch as th
import torch.nn as nn


class Detection(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Detection, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = "same")
        self.layer_center = nn.Conv2d(in_channels = self.out_channels, out_channels = 1, kernel_size = (1,1), padding ='same')
        self.layer_scale = nn.Conv2d(in_channels = self.out_channels, out_channels = 1,kernel_size = (1,1), padding = 'same')

    def forward(self, input):
        x = self.layer1(input)
        x = nn.functional.relu(x)
        center = self.layer_center(x)
        #center = th.nn.functional.interpolate(center, size=(1024,2048)) #(241,350) = input image size
        scale = self.layer_scale(x)
        #scale = th.nn.functional.interpolate(scale, size=(1024,2048))
        pred_center = th.sigmoid(center) #prediction heatmap
        pred_scale = nn.functional.relu(scale)
        pred_scale = th.log(pred_scale)
        return [pred_center, pred_scale]
