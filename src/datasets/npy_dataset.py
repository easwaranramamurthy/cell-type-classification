import numpy as np
import torch

class NpyDataset(torch.utils.data.Dataset):
    def __init__(self, X_path, Y_path):
        self.x = np.load(X_path, mmap_mode='r')
        self.y = np.load(Y_path, mmap_mode='r')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x.astype(int), y.astype(int)