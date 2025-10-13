import torch
import torch.nn as nn

class FlattenMLP(nn.Module):
    def __init__(self, in_dim: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, latent)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class VideoEncoder(nn.Module):
    def __init__(self, latent: int):
        super().__init__()
        # expects (B,T,C,H,W); we average time then MLP
        self.latent = latent
        self.mlp = None  # built at first call

    def forward(self, x):
        # x: (B,T,C,H,W) or (T,C,H,W) -> ensure batch
        if x.dim()==4:
            x = x.unsqueeze(0)
        x = x.mean(1)  # (B,C,H,W)
        B,C,H,W = x.shape
        if self.mlp is None:
            self.mlp = FlattenMLP(C*H*W, self.latent).to(x.device)
        return self.mlp(x)

class RadarEncoder(nn.Module):
    def __init__(self, in_ch: int, latent: int):
        super().__init__()
        self.latent = latent
        self.mlp = None
    def forward(self, x):
        if x.dim()==3:
            x = x.unsqueeze(0)
        B,C,H,W = x.shape
        if self.mlp is None:
            self.mlp = FlattenMLP(C*H*W, self.latent).to(x.device)
        return self.mlp(x)

class EEGEncoder(nn.Module):
    def __init__(self, chans: int, latent: int):
        super().__init__()
        self.mlp = None
        self.latent = latent
        self.chans = chans
    def forward(self, x):
        if x.dim()==2:
            x = x.unsqueeze(0)
        B,C,T = x.shape
        if self.mlp is None:
            self.mlp = FlattenMLP(C*T, self.latent).to(x.device)
        return self.mlp(x)

class FMRIEncoder(nn.Module):
    def __init__(self, latent: int):
        super().__init__()
        self.mlp = None
        self.latent = latent
    def forward(self, x):
        if x.dim()==4:
            x = x.unsqueeze(0)
        B,C,D,H,W = x.shape
        if self.mlp is None:
            self.mlp = FlattenMLP(C*D*H*W, self.latent).to(x.device)
        return self.mlp(x)
