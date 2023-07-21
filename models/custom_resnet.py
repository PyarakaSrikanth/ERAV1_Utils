import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class CustomRes_Network(nn.Module):
  def __init__(self):
    super(CustomRes_Network,self).__init__()
    self.prep = nn.Sequential(
        nn.Conv2d(3, 64 , 3 , 1 ,1),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    )
    self.layer1 = nn.Sequential(
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.MaxPool2d(2,2),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )
    self.residual1 = nn.Sequential(
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(128, 256, 3, 1, 1),
        nn.MaxPool2d(2,2),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )
    self.layer3 = nn.Sequential(
        nn.Conv2d(256, 512, 3, 1, 1),
        nn.MaxPool2d(2,2),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Dropout(0.1)
    )
    self.residual2 = nn.Sequential(
      nn.Conv2d(512, 512, 3, 1, 1),
      nn.ReLU(),
      nn.BatchNorm2d(512),
      nn.Conv2d(512, 512, 3, 1, 1),
      nn.ReLU(),
      nn.BatchNorm2d(512)
    )
    self.maxpool = nn.MaxPool2d(4,2)
    self.fc = nn.Linear(512,10)
    
    
  
  def forward(self, x):
    x = self.prep(x)
    residual1 = self.layer1(x)
    x = self.residual1(residual1)
    x += residual1
    x = self.layer2(x)
    residual2 = self.layer3(x)
    x = self.residual2(residual2)
    x += residual2
    x = self.maxpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return F.log_softmax(x,dim=-1)
