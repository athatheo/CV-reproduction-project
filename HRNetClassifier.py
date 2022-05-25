from hrnet_output import *


class HRNetClassifier(nn.Layer):
    '''校验完成：产生预测分类的结果，支持多分辨率预测输出

        params:
            inchannels:  输入大小
            num_classes: 分类数 > 0
    '''

    def __init__(self, inchannels, num_classes):
        super(HRNetClassifier, self).__init__()

        self.flatten = nn.Flatten()
        self.out_fc = nn.Linear(inchannels, num_classes)

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please make sure input data is list({0}) in HRNetOutput.".format(type(inputs))

        outs = []
        # 针对不同的输入，进行相应的操作
        for i in range(len(inputs)):
            out = inputs[i]
            out = self.flatten(out)
            out = self.out_fc(out)
            outs.append(out)

        return outs