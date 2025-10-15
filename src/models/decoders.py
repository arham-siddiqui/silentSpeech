import torch.nn as nn
class CTCHead(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, vocab_size)
    def forward(self, x):
        return self.proj(x)
