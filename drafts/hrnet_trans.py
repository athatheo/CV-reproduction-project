from hrnet_stage import *

class HRNetTrans(nn.Module):
    '''transform between stages
        transform the multi-branch output of last stage into the multi-branch input of the next stage
        params:
            old_branch_channels: the number of channels of a branch of last stage
            new_branch_channels: the number of channels of a branch of next stage

        eg:
           model = HRNetTrans([32], [32, 64])
           y1, y2 = model([tensor])
           output:
                [1, 32, 66, 66]
                [1, 64, 33, 33]
    '''

    def __init__(self, old_branch_channels, new_branch_channels):
        super(HRNetTrans, self).__init__()
        # check whether the transformation is from neighbour stages
        self.check_branch_num(old_branch_channels, new_branch_channels)

        # the number of branches from last stage
        self.old_branch_num = len(old_branch_channels)
        # the number of branches from next stage
        self.new_branch_num = len(new_branch_channels)

        # stage channel information
        self.old_branch_channels = old_branch_channels
        self.new_branch_channels = new_branch_channels

        # trans_layers layer assemble
        # contain two level of layers
        # outside -- index of original stage branches
        # inside -- index of new stage branches
        self.trans_layers = self.create_new_branch_trans_layers()

    def forward(self, inputs):
        assert isinstance(inputs, list), \
            "Please input list({0}) data in the trans layer.".format(type(inputs))
        outs = []

        # go through the information from last stage, output the contribution proportion to the next stage
        for i in range(0, self.old_branch_num):  # get the output from last stage
            x = inputs[i]  # the output from a branch from last stage
            # value of next stage branches
            out = []
            for j in range(0, self.new_branch_num):
                y = self.trans_layers[i][j](x)
                # print(i, '-', j, ' : ', self.new_branch_num, '--shape: ', y.shape)
                out.append(y)
            # fuse the contribution from last stage branches
            # copy the first results
            if len(outs) == 0:
                outs = out
            else:  # add the rest
                for i in range(self.new_branch_num):
                    outs[i] += out[i]

        # list with element tensor
        return outs

    def check_branch_num(self, old_branch_channels, new_branch_channels):
        '''
            check whether the branches between stages are reasonable
        '''
        assert len(new_branch_channels) - len(old_branch_channels) == 1, \
            "Please make sure the number of closed stage's branch is only less than 1."

    def create_new_branch_trans_layers(self):
        '''
            the network to produce next stage branches from last stage branches
        '''
        totrans_layers = []  # all the transform layers of all the branches

        # go through every branch of last stage
        for i in range(self.old_branch_num):
            branch_trans = []
            # produce elements of new branches
            for j in range(self.new_branch_num):
                if i == j:  # f(x) = x
                    layer = PlaceHolder()
                elif i > j:  # upsampling
                    layer = []
                    in_channels = self.old_branch_channels[i]
                    for k in range(i - j):
                        layer.append(
                            # change channel -- 1x1
                            nn.Conv2d(in_channels, self.new_branch_channels[j],
                                      kernel_size=1, bias=False)
                        )
                        layer.append(
                            # Batchnorm
                            nn.BatchNorm2d(self.new_branch_channels[j], momentum=BN_MOMENTUM)
                        )
                        layer.append(
                            # Activation function
                            nn.ReLU()
                        )
                        layer.append(
                            # Upsampling scale: 2.
                            nn.Upsample(scale_factor=2.)
                        )
                        in_channels = self.new_branch_channels[j]
                    layer = nn.Sequential(*layer)
                elif i < j:  # downsampling
                    layer = []
                    in_channels = self.old_branch_channels[i]
                    for k in range(j - i):
                        layer.append(
                            # change channel -- 1x1
                            nn.Conv2d(in_channels, self.new_branch_channels[j],
                                      kernel_size=1, bias=False)
                        )
                        layer.append(
                            # Downsampling scale: 1./2.
                            nn.Conv2d(self.new_branch_channels[j], self.new_branch_channels[j],
                                      kernel_size=3, stride=2, padding=1, bias=False)
                        )
                        layer.append(
                            # Batchnorm
                            nn.BatchNorm2d(self.new_branch_channels[j], momentum=BN_MOMENTUM)
                        )
                        layer.append(
                            # Activation function
                            nn.ReLU()
                        )
                        in_channels = self.new_branch_channels[j]
                    layer = nn.Sequential(*layer)
                branch_trans.append(layer)
            # build an output from original branch to new branch
            branch_trans = nn.ModuleList(branch_trans)
            totrans_layers.append(branch_trans)

        return nn.ModuleList(totrans_layers)