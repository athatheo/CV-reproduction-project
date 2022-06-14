import torch
from torch.nn import Module
from torch.nn import LeakyReLU
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn import CrossEntropyLoss


class FSH(Module):
    def __init__(self):
        super(FSH, self).__init__()
        self.num_classes = 2
        self.roi_feat_size = 7
        self.relu = LeakyReLU(0.2, inplace=True)
        self.conv1 = Conv2d(728, 256, kernel_size=3, padding=1)
        self.conv2 = Conv2d(256, 256, kernel_size=3, padding=1)
        last_layer_dim = 256*(self.roi_feat_size * self.roi_feat_size)
        self.linear1 = Linear(last_layer_dim, 1024)
        self.linear2 = Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear3 = Linear(1024, self.num_classes)
        self.init_weights()
        self.loss = CrossEntropyLoss()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        score = self.linear3(x)

        if self.num_classes > 1:
            return score
        return score.view(-1)

    def init_weights(self):
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)
        xavier_uniform_(self.linear3.weight)
        constant_(self.linear1.bias, 0)
        constant_(self.linear2.bias, 0)
        constant_(self.linear3.bias, 0)

    def loss(self, score, target):
        return self.loss(score, target)
