from hrnet_input import *
import torch
import numpy as np


# A simple test net


class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()

        self.input = HRNetInput(in_channels=3, out_channels=64, stage1_in_channels=32)

    def forward(self, inputs):
        x = self.input(inputs)

        return x


if __name__ == "__main__":
    model = TestNet()
    data = np.random.randint(0, 255, (1, 3, 256, 256)).astype(np.float32)
    data = torch.Tensor(data)
    y = model(data)
    print(y.shape)
