from hrnet_trans import *
class HRNetOutput(nn.Module):
    ''' output channel and pooling

        params:
            inchannels:  input channel
            outchannels: output channel
    '''

    def __init__(self, inchannels, outchannels):
        super(HRNetOutput, self).__init__()

        self.conv = nn.Conv2d(inchannels, outchannels, kernel_size=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)  # pooling
        self.relu = nn.ReLU()

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please make sure input data is list({0}) in HRNetOutput.".format(type(inputs))

        outs = []

        # different action according to different output
        for i in range(len(inputs)):
            out = inputs[i]
            out = self.conv(out)
            out = self.avgpool(out)
            out = self.relu(out)
            outs.append(out)

        return outs