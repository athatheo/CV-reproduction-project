# 1. Import dependencies
from torch import nn

# BatchNorm 2D
BN_MOMENTUM = 0.2


# 2. Placeholder, do f(x) = x, connect with no change
class PlaceHolder(nn.Module):

    def __init__(self):
        super(PlaceHolder, self).__init__()
        # produce nothing

    def forward(self, inputs):
        return inputs


# 3. Build the basic 3*3 conv because it is widely used
class HRNetConv3x3(nn.Module):
    """
    lots of 3x3 convolutions in the network, thus build it for later reuse
    """

    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super(HRNetConv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)

        return x


# 4. The main part before HRNet input
class HRNetStem(nn.Module):
    """
    Stem is used to process input pictures, transform them to feature map that can be the input of the main network
    **from paper**:
    The stem(two stride-2 3x3 convolutions decreasing the resolution to 1/4)
    ''
    """

    def __init__(self, in_channels, out_channels):
        super(HRNetStem, self).__init__()

        # resolution to 1/2
        self.conv1 = HRNetConv3x3(in_channels=in_channels, out_channels=out_channels, stride=2, padding=1)
        # resolution to 1/4
        self.conv2 = HRNetConv3x3(in_channels=out_channels, out_channels=out_channels, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)

        return x


# 5. Build input network
class HRNetInput(nn.Module):
    """
    # transform the output channel after stem to stage1
    in_channels: input features
    out_channels: the number of channels after stem
    stage1_channels: the input channel of stage1 module, corresponding to the width of HRNet
    """

    def __init__(self, in_channels, out_channels, stage1_in_channels):
        super(HRNetInput, self).__init__()
        self.stem = HRNetStem(in_channels=in_channels, out_channels=out_channels)
        self.in_change_conv = nn.Conv2d(in_channels=out_channels, out_channels=stage1_in_channels, kernel_size=1,
                                        stride=1, bias=False)
        self.in_change_bn = nn.BatchNorm2d(stage1_in_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.stem(inputs)
        x = self.in_change_conv(x)
        x = self.in_change_bn(x)
        x = self.relu(x)

        return x
# output torch.Size([1, 32, 64, 64]) : 1 sample, 32 channels, 64*64 resolution
