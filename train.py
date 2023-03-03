import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import transforms
from tqdm import tqdm
from model import VICReg
from loss import variance, invariance, covariance

class Augmentation:
    """
    Wrapper around a PyTorch transform, outputting two different augmentations
    for a single input. Applying this when loading a dataset ensures that a
    dataloader will provide two augmentations for each sample in a batch.
    """
    augment = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.RandomSolarize(0.5, p=0.2),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
        
    def __call__(self, x):
        return self.augment(x), self.augment(x)

# define model and move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder_dim, projector_dim = 512, 1024
model = VICReg(encoder_dim, projector_dim).to(device)

# prepare data, optimizer, and training hyperparams
num_epochs, batch_size = 500, 256
data = CIFAR10(root=".", train=True, download=True, transform=Augmentation())
dataloader = DataLoader(data, batch_size, shuffle=True, num_workers=2)
opt = Adam(model.parameters(), lr=2e-4, weight_decay=1e-6)
progress = tqdm(range(num_epochs))

# load from checkpoint if it exists
if os.path.exists("checkpoint.pt"):
    print("Loading from checkpoint...")
    cp = torch.load("checkpoint.pt")
    model.load_state_dict(cp["model_state_dict"])
    opt.load_state_dict(cp["optimizer_state_dict"])
    progress = tqdm(range(cp["epoch"], num_epochs))

# train the model and regularly save to disk
for epoch in progress:
    for images, _ in dataloader:
        x1, x2 = [x.to(device) for x in images]
        z1, z2 = model(x1, x2)
        
        la, mu, nu = 25, 25, 1
        var1, var2 = variance(z1), variance(z2)
        inv = invariance(z1, z2)
        cov1, cov2 = covariance(z1), covariance(z2)
        loss = la*inv + mu*(var1 + var2) + nu*(cov1 + cov2)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        progress.set_description(f"Loss: {loss.item()}")

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        torch.save({
            "epoch": epoch + 1,
            "encoder_dim": encoder_dim,
            "projector_dim": projector_dim,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict()
        },"checkpoint.pt")