import random
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from  torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def data_checker(dataloader):
    train_features, train_labels = next(iter(dataloader))
    
    indices = random.sample(range(len(train_features)), 9)
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img = train_features[idx].squeeze()
        label = train_labels[idx].item()
        axes[i].set_title(f"{labels_map[label]}")
        axes[i].imshow(img, cmap="gray")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    

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
        

def train(model, dataloader, optimizer, criterion, epoch, total_epochs):
    model.train()
    train_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')
    
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outs = model(features)
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * features.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_loss = train_loss / len(dataloader.dataset)
    print('Epoch: {}/{} \tTraining Loss: {:.6f}'.format(epoch, total_epochs, train_loss))
        

def val(model, dataloader, criterion, epoch, total_epochs):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [Val]')
    
    with torch.no_grad():
        for features, labels in progress_bar:
            features = features.to(device)
            labels = labels.to(device)
            outs = model(features)
            loss = criterion(outs, labels)
            val_loss += loss.item() * features.size(0)
            
            # 计算准确率
            _, predicted = torch.max(outs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条显示当前的loss和accuracy
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})
        
    val_loss = val_loss / len(dataloader.dataset)
    accuracy = 100 * correct / total
    print(f'Epoch: {epoch+1}/{total_epochs} \tVal Loss: {val_loss:.6f} \tAccuracy: {accuracy:.2f}%')


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
    
    batch_size = 1024
    lr = 1e-4
    epochs = 20
    
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    
    
    train_datasets = FashionMNISTDataset("../datasets/FashionMNIST/fashion-mnist_train.csv", transform=transform)
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=False)
    
    test_datasets = FashionMNISTDataset("../datasets/FashionMNIST/fashion-mnist_train.csv", transform=transform)
    test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=False)
    
    ## 检查数据集是否正确加载
    # data_checker(train_loader)
    # data_checker(test_loader)
    
    model = NetWork().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, epoch, epochs)
        val(model, test_loader, criterion, epoch, epochs)
    
    
    