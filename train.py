import torch

from dataload import CityPersons
from torch.utils.data import DataLoader
from dataload import Config

config = Config()

cp = CityPersons(path='datasets/CityPersons', mode='train', config=config)

loader = DataLoader(cp, batch_size=8, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.cuda()
model.train()
print(len(loader))


for batch in loader:
    inputs, labels = batch
    inputs = inputs.to(device=device)
