from hrnet_trans import *

# keep: only output
# fuse: fuse to high resolution
# multi: multi output
Fusion_Mode = ['keep', 'fuse', 'multi']


class HRNetFusion(nn.Module):
    """

        params:
            stage4_channels: list -- list of the branch channels
            mode: fuse mode
    """

    def __init__(self, stage4_channels, mode='keep'):
        super(HRNetFusion, self).__init__()

        assert mode in Fusion_Mode, \
            "Please make sure mode({0}) is in ['keep', 'fuse', 'multi'].".format(mode)

        self.stage4_channels = stage4_channels
        self.mode = mode

        # fuse layer
        self.fuse_layers = self.create_fusion_layers()

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please make sure input data is list({0}) in HRNetOutput.".format(type(inputs))

        # get inputs from different branches
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
        """
            build appropriate fuse layers
        """
        layer = None

        if self.mode == Fusion_Mode[0]:  # keep
            layer = self.create_keep_fusion_layers()
        elif self.mode == Fusion_Mode[1]:  # fuse
            layer = self.create_fuse_fusion_layers()
        elif self.mode == Fusion_Mode[2]:  # multi
            layer = self.create_multi_fusion_layers()

        return layer

    def create_keep_fusion_layers(self):
        """a
            highest resolution
        """
        self.outchannels = self.stage4_channels[0]
        return PlaceHolder()

    def create_fuse_fusion_layers(self):
        """b
            fuse different resolutions, keep largest channel
        """
        layers = []
        outchannel = self.stage4_channels[3]  # keep max channel

        # go through channels
        for i in range(0, len(self.stage4_channels)):
            inchannel = self.stage4_channels[i]  # get number of channels of each branch
            layer = []

            # not first branch, transform channels
            if i != len(self.stage4_channels) - 1:
                layer.append(nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False))
                layer.append(nn.BatchNorm2d(outchannel, momentum=BN_MOMENTUM))
                layer.append(nn.ReLU())

            # upsamling according to current index
            for j in range(i):
                layer.append(nn.Upsample(scale_factor=2.))

            layer = nn.Sequential(*layer)
            layers.append(layer)

        self.outchannels = outchannel
        return nn.ModuleList(layers)

    def create_multi_fusion_layers(self):
        '''c
            fuse multiple layers output multiple resolutions
        '''
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

        # downsample of other resolutions
        # branch1 => branch2
        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(outchannel, momentum=BN_MOMENTUM),
                nn.ReLU()
            )
        )
        # branch2 => branch3
        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(outchannel, momentum=BN_MOMENTUM),
                nn.ReLU()
            )
        )
        # branch3 => branch4
        multi_fuse_layers.append(
            nn.Sequential(
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(outchannel, momentum=BN_MOMENTUM),
                nn.ReLU()
            )
        )

        self.outchannels = outchannel  # current channels
        return nn.ModuleList(multi_fuse_layers)