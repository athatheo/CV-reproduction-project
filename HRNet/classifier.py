from torch import nn
from model import HRNet


class HRNetPoolOutput(nn.Module):

    def __init__(self, inchannels, outchannels):
        super(HRNetPoolOutput, self).__init__()

        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please make sure input data is list({0}) in HRNetOutput.".format(type(inputs))

        outs = []

        # acts according to different outputs
        for j in range(len(inputs)):
            out = inputs[j]
            out = self.conv(out)
            out = self.avgpool(out)
            out = self.relu(out)
            outs.append(out)

        return outs


class HRNetClassifier(nn.Module):

    def __init__(self, inchannels, num_classes):
        super(HRNetClassifier, self).__init__()

        self.flatten = nn.Flatten()
        self.out_fc = nn.Linear(inchannels, num_classes)

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please make sure input data is list({0}) in HRNetOutput.".format(type(inputs))

        outs = []
        # acts according to different outputs
        for j in range(len(inputs)):
            out = inputs[j]
            out = self.flatten(out)
            out = self.out_fc(out)
            outs.append(out)

        return outs


class HRNetClassification(nn.Module):

    def __init__(self, num_classes, width=32, mode='fuse'):
        super(HRNetClassification, self).__init__()

        self.mode = mode
        self.width = width

        # backbone network
        self.hrnet = HRNet(width=width, mode=mode)
        # pooling
        self.output = HRNetPoolOutput(self.hrnet.hrnet_fusion.outchannels[0], 2048)
        # classifier
        self.classifier = HRNetClassifier(2048, num_classes)

    def forward(self, inputs):

        x = self.hrnet(inputs)
        x = self.output(x)
        x = self.classifier(x)  # return list

        if self.mode == 'multi':
            return [x[-1]]
        else:
            return [x[0]]