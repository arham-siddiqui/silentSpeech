import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, latent: int):
        super().__init__()
        self.q = nn.Linear(latent, latent, bias=False)
        self.k = nn.Linear(latent, latent, bias=False)
        self.v = nn.Linear(latent, latent, bias=False)
    def forward(self, zs):
        # zs: list of (B,latent)
        Z = torch.stack(zs, dim=1)  # (B,M,L)
        Q = self.q(Z); K = self.k(Z); V = self.v(Z)
        att = torch.softmax((Q@K.transpose(-1,-2))/ (Z.size(-1)**0.5), dim=-1)
        FZ = (att@V).mean(1)
        return FZ

def info_nce(zs, temperature: float=0.07):
    # simple multi-view InfoNCE over batch
    # zs: list of (B,L)
    if len(zs) < 2:
        return zs[0].pow(2).mean()
    B = zs[0].size(0)
    Z = torch.stack([F.normalize(z, dim=-1) for z in zs], dim=1)  # (B,M,L)
    # use first two views only for loss (simple baseline)
    z1, z2 = Z[:,0,:], Z[:,1,:]
    logits = (z1 @ z2.t()) / temperature
    labels = torch.arange(B, device=logits.device)
    return F.cross_entropy(logits, labels)
