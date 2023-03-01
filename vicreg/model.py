import torch
from torch import nn
from torchvision.models.resnet import resnet18

class Projector(nn.Module):
    def __init__(self, encoder_dim, projector_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(encoder_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.BatchNorm1d(projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim)
        )
            
    def forward(self, x):
        return self.network(x)
    
class VICReg(nn.Module):
    def __init__(self, encoder_dim, projector_dim):
        super().__init__()
        self.encoder = resnet18(num_classes=encoder_dim)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=1)
        self.encoder.maxpool = nn.Identity()
        self.projector = Projector(encoder_dim, projector_dim)
    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2))
        y = self.encoder(x)
        return self.projector(y).chunk(2)
