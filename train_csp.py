import torch

from dataload import CityPersons
from torch.utils.data import DataLoader
from dataload import Config
from pathlib import Path

from HRNet_pretrain.hrnetpretrain1 import hrnetv2_32
from CSP_Head import csp_head, losses_head

import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

def training():
    transform = T.ToPILImage()


    #from HRNet_pretrain.hrnetpretrain import hrnetv2_32
    config = Config()

    #import dataset : you may have to change the path
    cp = CityPersons(path='D:/datasets/CityPersons', mode='train', config=config)
    loader = DataLoader(cp, batch_size=4, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # creation model :
    backbone = hrnetv2_32(pretrained = True)

    #freeze the backbone parameters :
    for para in backbone.parameters():
        para.requires_grad = False

    learning_rate = 0.0002
    csp = csp_head.Detection(in_channels = 480, out_channels = 256, kernel_size = 3 )
    optimizer = torch.optim.Adam(csp.parameters(), lr=learning_rate)
    loss_function = losses_head.loss

    # you can uncomment the lignes below to load the pre trained csp model
    #PATH = "checkpoint_loss.pt" #for saving the models

    #checkpoint = torch.load(PATH, map_location = torch.device('cpu'))
    #csp.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #Loss = checkpoint['loss']

    num_epochs = 4
    # to save the trained parameters
    PATH = "checkpoint_neww_losss.pt"


    csp.train()
    Loss = []
    for epoch in range(num_epochs):
        loss_epoch = []
        for i, batch in enumerate(loader):


            inputs, labels = batch
            #print(inputs.shape)
            inputs = inputs.to(device=device)
            input_csp = backbone.forward(inputs)
            output = csp.forward(input_csp)
            #print(output[0].shape)
            #img = transform(output[1][0,0,:,:])
            #img.show()
            #plt.imshow(output[0][0,0,:,:].detach().numpy(), cmap='hot', interpolation='nearest')
            #plt.show()
            #img = transform(output[0][0,0,:,:])
            #img.show()
            img = transform(inputs[0][:,:,:])
            img.show()
            optimizer.zero_grad()
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss)
            Loss.append(loss)
        
            


torch.save({
                'epoch': epoch,
                'i': i,
                'model_state_dict': csp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': Loss,
                }, PATH)


if __name__ == "__main__":
    training()