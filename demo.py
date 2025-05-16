import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from  torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class FashionMNISTDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.imgs = pd.read_csv(path).iloc[:, 1:].values.astype(np.uint8)
        self.labels = pd.read_csv(path).iloc[:, 0].values
        self.transform = transform
    
    def __len__(self):
        if len(self.imgs) != len(self.labels):
            raise ValueError(f"images len: {len(self.imgs)} not equals to labels len: {len(self.labels)}")
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx].reshape(28, 28, -1)
        label = int(self.labels[idx])
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.tensor(img / 255., dtype=torch.float)
            
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label
    
    
class NetWork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.flatten = nn.Flatten()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=128 * 3 * 3, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=10)
        )
        
    def forward(self, x):
        features = self.conv(x)
        features = self.flatten(features)
        logits = self.fc(features)
        
        return logits
        

if __name__ == "__main__":
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 256
    lr = 1e-4
    epochs = 20
    
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    
    
    train_datasets = FashionMNISTDataset("./datasets/FashionMNIST/fashion-mnist_train.csv", transform=transform)
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=False)
    
    test_datasets = FashionMNISTDataset("./datasets/FashionMNIST/fashion-mnist_train.csv", transform=transform)
    test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=False)
    
    ## 检查数据集是否正确加载
    # train_features, train_labels = next(iter(train_loader))
    
    # indices = random.sample(range(len(train_features)), 9)
    # fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    # axes = axes.flatten()
    
    # for i, idx in enumerate(indices):
    #     img = train_features[idx].squeeze()
    #     label = train_labels[idx].item()
    #     axes[i].set_title(f"{labels_map[label]}")
    #     axes[i].imshow(img, cmap="gray")
    #     axes[i].axis('off')
    
    # plt.tight_layout()
    # plt.show()