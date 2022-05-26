from hrnet_fusion import *
class HRNet(nn.Module):
    """ 2048-1x1
        HRNet （WIDTH=32）, expand using [32, 64, 128, 256]

        V1   OUTPUT: maximum resolution[64x64] -- unchanged channels  -- mode:keep
        V2   OUTPUT: fuse maximum resolution[64x64] -- expanded channels(256) -- mode:fuse
        V2P  OUTPUT: multi kind resolution[64x64, 32x32, 16x16, 8x8] -- expanded channels(256) -- mode:multi

        > output did not pass softmax

        params:
            num_classes: number of classes, if 0，output 2048-1x1 feature map
            mode:        head fuse output
                         keep： directly output maximum resolution
                         fuse:  fuse maximum resolution
                         multi: multi kind resolution
    """

    def __init__(self, num_classes=2, mode='fuse'):
        super(HRNet, self).__init__()
        self.num_classes = num_classes
        self.width = 32  # -- width

        self.n_stage_channels = [[32], [32, 64], [32, 64, 128], [32, 64, 128, 256]]

        # input of hrnet -- 3 to 64 to 32
        self.hrnet_input = HRNetInput(in_channels=3, out_channels=64, stage1_inchannels=32)  # OUTPUT WIDTH 64

        # stage1 feature extraction -- supporting block：NormalBlock, ResidualBlock
        self.stage1 = HRNetStage(stage_channels=[32], block=ResidualBlock)
        self.trans_stage1to2 = HRNetTrans([32], [32, 64])

        # stage2 feature extraction
        self.stage2 = HRNetStage(stage_channels=[32, 64], block=ResidualBlock)
        self.trans_stage2to3 = HRNetTrans([32, 64], [32, 64, 128])

        # stage3 feature extraction
        self.stage3 = HRNetStage(stage_channels=[32, 64, 128], block=ResidualBlock)
        self.trans_stage3to4 = HRNetTrans([32, 64, 128], [32, 64, 128, 256])

        # stage4 feature extraction
        self.stage4 = HRNetStage(stage_channels=[32, 64, 128, 256], block=ResidualBlock)

        # output head after stage 4
        self.hrnet_fusion = HRNetFusion([32, 64, 128, 256], mode=mode)

        # last output layer
        self.output = HRNetOutput(self.hrnet_fusion.outchannels, 2048)

        # classification judge
        if num_classes != 0:
            self.classifier = HRNetClassifier(2048, num_classes)

    def forward(self, inputs):
        x = self.hrnet_input(inputs)
        x = [x]  # list

        # stage1
        x = self.stage1(x)  # return x  list
        x = self.trans_stage1to2(x)  # branch fuse expand 1-2

        # stage2
        x = self.stage2(x)
        x = self.trans_stage2to3(x)  # branch fuse expand 2-3

        # stage3
        x = self.stage3(x)
        x = self.trans_stage3to4(x)  # branch fuse expand 3-4

        # stage4
        x = self.stage4(x)

        # representation head  -- output fuse layer
        x = self.hrnet_fusion(x)  # return x list

        # expand channel output layer
        x = self.output(x)  # return x list

        # classification prediction
        if self.num_classes != 0:
            x = self.classifier(x)  # return x list

        return x