from torch.utils.data import DataLoader, Dataset
import os 
import torchvision 
import numpy as np 
import matplotlib.pyplot as plt 

class CIFAR10(Dataset):
    def __init__(self,root_dir,test=False):
        self.root_dir = root_dir
        if test:
            self.path = os.path.join(self.root_dir,"Data","cifar10","test")
        else:
            self.path = os.path.join(self.root_dir,"Data","cifar10","train")

    
    def __len__(self):
        return len(os.listdir(self.path))

    def __getitem__(self,idx):
        img_path = os.path.join(self.path,os.listdir(self.path)[idx])

        return img_path

train_set = CIFAR10(root_dir=r"D:\Artificial Intelligence\CNN\CIFAR10")
train_loader = DataLoader(train_set,batch_size=8,shuffle=True)

data = iter(train_loader)
print(next(data))
