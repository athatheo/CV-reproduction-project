from hrnet_trans import *

class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()
        self.input = HRNetInput(in_channels=3, out_channels=64, stage1_inchannels=32)

        self.stage1 = HRNetStage([32], NormalBlock)
        self.trans1 = HRNetTrans([32], [32, 64])

        self.stage2 = HRNetStage([32, 64], NormalBlock)
        self.trans2 = HRNetTrans([32, 64], [32, 64, 128])

        self.stage3 = HRNetStage([32, 64, 128], NormalBlock)
        self.trans3 = HRNetTrans([32, 64, 128], [32, 64, 128, 256])

        self.stage4 = HRNetStage([32, 64, 128, 256], NormalBlock)


    def forward(self, inputs):

        x = self.input(inputs)
        x = [x]

        x = self.stage1(x)
        x = self.trans1(x)

        x = self.stage2(x)
        x = self.trans2(x)

        x = self.stage3(x)
        x = self.trans3(x)

        x = self.stage4(x)
        return x

if __name__ == "__main__":
    model = TestNet()
    data = np.random.randint(0, 255, (1, 3, 256, 256)).astype(np.float32)
    data = torch.Tensor(data)
    y = model(data)
    for i in range(len(y)):
        print(y[i].shape)

# torch.Size([1, 32, 64, 64])   C -- 1/4
# torch.Size([1, 64, 32, 32])   2C -- 1/4 * 1/2 = 1/8
# torch.Size([1, 128, 16, 16])  4C -- 1/8 * 1/2 = 1/16
# torch.Size([1, 256, 8, 8])    8C -- 1/16 * 1/2 = 1/32
'''
From paper: The resolutions are 1/4, 1/8, 1/16 and 1/32.
'''