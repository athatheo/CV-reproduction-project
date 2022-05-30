import logging
from torch import nn
from checkpoint import load_checkpoint

# BatchNorm momentum
BN_MOMENTUM = 0.2


class HRNetConv3x3(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super(HRNetConv3x3, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)

        return x


class HRNetStem(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(HRNetStem, self).__init__()

        self.conv1 = HRNetConv3x3(in_channels=in_channels, out_channels=out_channels, stride=2, padding=1)
        self.conv2 = HRNetConv3x3(in_channels=out_channels, out_channels=out_channels, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)

        return x


class PlaceHolder(nn.Module):

    def __init__(self):
        super(PlaceHolder, self).__init__()

    def forward(self, inputs):
        return inputs


class HRNetTrans(nn.Module):

    def __init__(self, old_branch_channels, new_branch_channels):
        super(HRNetTrans, self).__init__()

        self.check_branch_num(old_branch_channels, new_branch_channels)

        self.old_branch_num = len(old_branch_channels)

        self.new_branch_num = len(new_branch_channels)

        self.old_branch_channels = old_branch_channels
        self.new_branch_channels = new_branch_channels

        self.trans_layers = self.create_new_branch_trans_layers()

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please input list({0}) data in the trans layer.".format(type(inputs))
        outs = []

        for j in range(0, self.old_branch_num):  # output of last stage
            x = inputs[j]

            out = []
            for k in range(0, self.new_branch_num):
                y = self.trans_layers[j][k](x)
                # print(i, '-', j, ' : ', self.new_branch_num, '--shape: ', y.shape)
                out.append(y)

            if len(outs) == 0:
                outs = out
            else:
                for k in range(self.new_branch_num):
                    outs[k] += out[k]

        return outs

    def check_branch_num(self, old_branch_channels, new_branch_channels):

        assert len(new_branch_channels) - len(old_branch_channels) == 1, \
            "Please make sure the number of closed stage's branch is only less than 1."

    def create_new_branch_trans_layers(self):

        totrans_layers = []

        for i in range(self.old_branch_num):
            branch_trans = []

            for j in range(self.new_branch_num):
                if i == j:
                    layer = PlaceHolder()
                elif i > j:
                    layer = []
                    inchannels = self.old_branch_channels[i]
                    for k in range(i - j):
                        layer.append(

                            nn.Conv2d(inchannels, self.new_branch_channels[j],
                                      kernel_size=1, bias=False)
                        )
                        layer.append(

                            nn.BatchNorm2d(self.new_branch_channels[j], momentum=BN_MOMENTUM)
                        )
                        layer.append(

                            nn.ReLU()
                        )
                        layer.append(

                            nn.Upsample(scale_factor=2.)
                        )
                        inchannels = self.new_branch_channels[j]
                    layer = nn.Sequential(*layer)
                elif i < j:
                    layer = []
                    inchannels = self.old_branch_channels[i]
                    for k in range(j - i):
                        layer.append(

                            nn.Conv2d(inchannels, self.new_branch_channels[j],
                                      kernel_size=1, bias=False)
                        )
                        layer.append(

                            nn.Conv2d(self.new_branch_channels[j], self.new_branch_channels[j],
                                      kernel_size=3, stride=2, padding=1, bias=False)
                        )
                        layer.append(

                            nn.BatchNorm2d(self.new_branch_channels[j], momentum=BN_MOMENTUM)
                        )
                        layer.append(

                            nn.ReLU()
                        )
                        inchannels = self.new_branch_channels[j]
                    layer = nn.Sequential(*layer)
                branch_trans.append(layer)
            branch_trans = nn.ModuleList(branch_trans)
            totrans_layers.append(branch_trans)

        return nn.ModuleList(totrans_layers)


class NormalBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(NormalBlock, self).__init__()

        self.conv1 = HRNetConv3x3(in_channels=in_channels, out_channels=out_channels, stride=1, padding=1)
        self.conv2 = HRNetConv3x3(in_channels=out_channels, out_channels=out_channels, stride=1, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = HRNetConv3x3(in_channels=in_channels, out_channels=out_channels, padding=1)

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


class HRNetInput(nn.Module):

    def __init__(self, in_channels, out_channels, stage1_inchannels):
        super(HRNetInput, self).__init__()

        self.stem = HRNetStem(in_channels, out_channels)  # OUTPUT WIDTH 64

        self.in_change_conv = nn.Conv2d(in_channels=out_channels, out_channels=stage1_inchannels, kernel_size=1,
                                        stride=1, bias = False)
        self.in_change_bn = nn.BatchNorm2d(stage1_inchannels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.stem(inputs)
        x = self.in_change_conv(x)
        x = self.in_change_bn(x)
        x = self.relu(x)

        return x


class HRNetStage(nn.Module):

    def __init__(self, stage_channels, block):
        super(HRNetStage, self).__init__()

        assert isinstance(stage_channels, list), \
            "Please make sure the stage_channels type is list({0}) in HRNetStage".format(type(stage_channels))

        self.stage_channels = stage_channels
        self.stage_branch_num = len(stage_channels)
        self.block = block
        self.block_num = 4
        self.stage_layers = self.create_stage_layers()

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please make sure input data is list({0}) in the HRNetStage forward".format(type(inputs))

        outs = []

        for i in range(self.stage_branch_num):
            x = inputs[i]
            y = self.stage_layers[i](x)
            outs.append(y)

        return outs

    def create_stage_layers(self):

        tostage_layers = []

        for i in range(self.stage_branch_num):
            branch_layer = []
            for j in range(self.block_num):
                branch_layer.append(self.block(self.stage_channels[i], self.stage_channels[i]))
            branch_layer = nn.Sequential(*branch_layer)
            tostage_layers.append(branch_layer)

        return nn.ModuleList(tostage_layers)


# keep:
# fuse:
# multi:
Fusion_Mode = ['keep', 'fuse', 'multi']


class HRNetFusion(nn.Module):

    def __init__(self, stage4_channels, mode='keep'):
        super(HRNetFusion, self).__init__()

        self.outchannels = None
        assert mode in Fusion_Mode, \
            "Please make sure mode({0}) is in ['keep', 'fuse', 'multi'].".format(mode)

        self.stage4_channels = stage4_channels
        self.mode = mode

        self.fuse_layers = self.create_fusion_layers()

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please make sure input data is list({0}) in HRNetOutput.".format(type(inputs))

        x1, x2, x3, x4 = inputs
        outs = []

        if self.mode == Fusion_Mode[0]:  # keep
            out = self.fuse_layers(x1)
            outs.append(out)
        elif self.mode == Fusion_Mode[1]:  # fuse
            out = self.fuse_layers[0](x1)
            out += self.fuse_layers[1](x2)
            out += self.fuse_layers[2](x3)
            out += self.fuse_layers[3](x4)
            outs.append(out)
        elif self.mode == Fusion_Mode[2]:  # multi
            out1 = self.fuse_layers[0][0](x1)
            out1 += self.fuse_layers[0][1](x2)
            out1 += self.fuse_layers[0][2](x3)
            out1 += self.fuse_layers[0][3](x4)
            outs.append(out1)
            out2 = self.fuse_layers[1](out1)
            outs.append(out2)
            out3 = self.fuse_layers[2](out2)
            outs.append(out3)
            out4 = self.fuse_layers[3](out3)
            outs.append(out4)

        return outs

    def create_fusion_layers(self):

        layer = None

        if self.mode == Fusion_Mode[0]:  # keep
            layer = self.create_keep_fusion_layers()
        elif self.mode == Fusion_Mode[1]:  # fuse
            layer = self.create_fuse_fusion_layers()
        elif self.mode == Fusion_Mode[2]:  # multi
            layer = self.create_multi_fusion_layers()

        return layer

    def create_keep_fusion_layers(self):

        self.outchannels = [self.stage4_channels[0]] * 4  # list

    def create_fuse_fusion_layers(self):

        layers = []
        outchannel = self.stage4_channels[3]

        for i in range(0, len(self.stage4_channels)):
            inchannel = self.stage4_channels[i]
            layer = []

            if i != len(self.stage4_channels) - 1:
                layer.append(nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False))
                layer.append(nn.BatchNorm2d(outchannel, momentum=BN_MOMENTUM))
                layer.append(nn.ReLU())

            for j in range(i):
                layer.append(nn.Upsample(scale_factor=2.))

            layer = nn.Sequential(*layer)
            layers.append(layer)

        self.outchannels = [outchannel] * 4  # list
        return nn.ModuleList(layers)

    def create_multi_fusion_layers(self):

        multi_fuse_layers = []

        max_resolution_fuse_layers = []
        outchannel = self.stage4_channels[3]  # keep max channel

        for i in range(0, len(self.stage4_channels)):
            inchannel = self.stage4_channels[i]
            layer = []

            if i != len(self.stage4_channels) - 1:
                layer.append(nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False))
                layer.append(nn.BatchNorm2d(outchannel, momentum=BN_MOMENTUM))
                layer.append(nn.ReLU())

            for j in range(i):
                layer.append(nn.Upsample(scale_factor=2.))

            layer = nn.Sequential(*layer)
            max_resolution_fuse_layers.append(layer)
        max_resolution_fuse_layers = nn.ModuleList(max_resolution_fuse_layers)
        multi_fuse_layers.append(max_resolution_fuse_layers)  # branch1

        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(outchannel, momentum=BN_MOMENTUM),
                nn.ReLU()
            )
        )

        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(outchannel, momentum=BN_MOMENTUM),
                nn.ReLU()
            )
        )

        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(outchannel, momentum=BN_MOMENTUM),
                nn.ReLU()
            )
        )

        self.outchannels = [outchannel] * 4
        return nn.ModuleList(multi_fuse_layers)


class HRNet(nn.Module):

    def __init__(self, width=32, mode='fuse'):
        super(HRNet, self).__init__()
        self.width = width

        if self.width == 16:
            self.n_stage_channels = [[16], [16, 32], [16, 32, 64], [16, 32, 64, 128]]
        elif self.width == 32:
            self.n_stage_channels = [[32], [32, 64], [32, 64, 128], [32, 64, 128, 256]]
        elif self.width == 64:
            self.n_stage_channels = [[64], [64, 128], [64, 128, 256], [64, 128, 256, 512]]
        elif self.width == 128:
            self.n_stage_channels = [[128], [128, 256], [128, 256, 512], [128, 256, 512, 1024]]

        self.hrnet_input = HRNetInput(in_channels=3, out_channels=64, stage1_inchannels=self.width)

        self.stage1 = HRNetStage(stage_channels=self.n_stage_channels[0], block=ResidualBlock)
        self.trans_stage1to2 = HRNetTrans(self.n_stage_channels[0], self.n_stage_channels[1])

        self.stage2 = HRNetStage(stage_channels=self.n_stage_channels[1], block=ResidualBlock)
        self.trans_stage2to3 = HRNetTrans(self.n_stage_channels[1], self.n_stage_channels[2])

        self.stage3 = HRNetStage(stage_channels=self.n_stage_channels[2], block=ResidualBlock)
        self.trans_stage3to4 = HRNetTrans(self.n_stage_channels[2], self.n_stage_channels[3])

        self.stage4 = HRNetStage(stage_channels=self.n_stage_channels[3], block=ResidualBlock)

        self.hrnet_fusion = HRNetFusion(self.n_stage_channels[3], mode=mode)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, inputs):
        x = self.hrnet_input(inputs)
        x = [x]

        x = self.stage1(x)
        x = self.trans_stage1to2(x)

        x = self.stage2(x)
        x = self.trans_stage2to3(x)

        x = self.stage3(x)
        x = self.trans_stage3to4(x)

        x = self.stage4(x)

        x = self.hrnet_fusion(x)

        return x
