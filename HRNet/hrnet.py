import torch
import numpy as np
from classifier import HRNetClassification
from model import HRNet

if __name__ == "__main__":
    model = HRNet(width=32, mode='fuse')
    # model = HRNetClassification(num_classes=2, width=32, mode='multi')
    data = np.random.randint(0, 256, (1, 3, 256, 256)).astype(np.float32)
    data = torch.Tensor(data)

    y_preds = model(data)

    for i in range(len(y_preds)):
        print(y_preds[i].shape)