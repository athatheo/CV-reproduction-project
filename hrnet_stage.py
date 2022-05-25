from hrnet_input import *


# 6. Build the blocks used in stages
class NormalBlock(nn.Module):
    """
        'Each branch in multi-resolution parallel convolution of the modularized block contains 4 residual units.
        Each unit contains two 3 x 3 convolutions for each resolution, where each convolution is followed by batch
        normalization and the nonlinear activation ReLU.'
        params:
            in_channels:  number of input channels
            out_channels: number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super(NormalBlock, self).__init__()

        self.conv1 = HRNetConv3x3(in_channels=in_channels, out_channels=out_channels, stride=1, padding=1)
        self.conv2 = HRNetConv3x3(in_channels=out_channels, out_channels=out_channels, stride=1, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)

        return x


class ResidualBlock(nn.Module):
    """
        params:
            in_channels:  number of input channels
            out_channels: number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = HRNetConv3x3(in_channels=in_channels, out_channels=out_channels, padding=1)
        # Batch-norm and relu should be activated after the residual
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        residual = inputs

        x = self.conv1(inputs)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x


# 7. Build the stage in HRNet
# Sequential
# Module-list Module[0](Sequential), Module[1](Sequential)
class HRNetStage(nn.Module):
    """
    params:
        stage_channels: the list of channels in certain branch in current stage (e.g. stage 4 [32, 64, 128, 256])
        block: the type of every block in each branch
    """

    def __init__(self, stage_channels, block):
        super(HRNetStage, self).__init__()

        assert isinstance(stage_channels, list), \
            "Please make sure the stage_channels type is list({0}) in HRNetStage".format(type(stage_channels))

        self.stage_channels = stage_channels
        self.stage_branch_num = len(stage_channels)
        self.block = block
        self.block_num = 4

        # get all the layers of certain stage
        self.stage_layers = self.create_stage_layers()

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please make sure input data is list({0}) in the HRNetStage forward".format(type(inputs))

        outs = []

        # go through every branch
        for i in range(self.stage_branch_num):
            x = inputs[i]
            y = self.stage_layers[i](x)  # pass corresponding path
            outs.append(y)

        return outs

    def create_stage_layers(self):
        """
            first create sequential structure then make a parallel for one single stage
        """
        stage_layers = []  # Parallel structure

        # go through every branch in one stage for sequential structure
        for i in range(self.stage_branch_num):
            branch_layer = []  # Sequential structure
            # every branch has 4 blocks
            for j in range(self.block_num):
                branch_layer.append(self.block(self.stage_channels[i], self.stage_channels[i]))
            branch_layer = nn.Sequential(*branch_layer)
            # put in the stage
            stage_layers.append(branch_layer)
        # parallel module list
        return nn.ModuleList(stage_layers)
