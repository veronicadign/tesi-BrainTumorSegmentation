import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, img_dir, img_list, mask_dir, mask_list):
        self.img_dir = img_dir
        self.img_list = img_list
        self.mask_dir = mask_dir
        self.mask_list = mask_list
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_list[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_list[idx])
        
        image = np.load(img_name)
        mask = np.load(mask_name)
        
        return torch.from_numpy(image), torch.from_numpy(mask)


def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    dataset = CustomDataset(img_dir, img_list, mask_dir, mask_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader