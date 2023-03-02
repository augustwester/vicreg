import torch
from torch.nn.functional import mse_loss, relu

def variance(z, gamma=1):
    return relu(gamma - z.std(0)).mean()

def invariance(z1, z2):
    return mse_loss(z1, z2)

def covariance(z):
    n, d = z.shape
    mu = z.mean(0)
    cov = torch.einsum("ni,nj->ij", z-mu, z-mu) / (n - 1)
    off_diag = cov.pow(2).sum() - cov.pow(2).diag().sum()
    return off_diag / d
