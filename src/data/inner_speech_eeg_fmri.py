from torch.utils.data import Dataset
class InnerSpeechEEGfMRI(Dataset):
    def __init__(self, root, subjects):
        self.n=8
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        import torch
        return {'eeg': torch.randn(64,100), 'fmri': torch.randn(1,8,8,8)}
