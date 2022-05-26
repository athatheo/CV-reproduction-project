from hrnet_stage import *
import torch
import numpy as np

class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()
        self.input = HRNetInput(in_channels=3, out_channels=64, stage1_inchannels=32)
        self.stage1 = HRNetStage([32], NormalBlock)

    def forward(self, inputs):
        x = self.input(inputs)
        x = [x]
        x = self.stage1(x)
        return x


if __name__ == "__main__":
    model = TestNet()
    data = np.random.randint(0, 255, (1, 3, 256, 256)).astype(np.float32)
    data = torch.Tensor(data)
    y = model(data)
    print(y[0].shape)
