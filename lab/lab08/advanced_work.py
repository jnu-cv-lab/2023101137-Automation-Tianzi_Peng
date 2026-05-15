import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(dataset_name='MNIST', batch_size=64):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_class = torchvision.datasets.MNIST
        in_channels = 1
        img_size = 28
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_class = torchvision.datasets.CIFAR10
        in_channels = 3
        img_size = 32
    else:
        raise ValueError("不支持的数据集")

    full_train = dataset_class(root='./data', train=True, download=True, transform=transform)
    test_data = dataset_class(root='./data', train=False, download=True, transform=transform)
    
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_data, val_data = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, in_channels, img_size

class BaseCNN(nn.Module):
    """基础结构：对应基础任务的浅层网络"""
    def __init__(self, in_channels, img_size):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc_size = 32 * (img_size // 4) * (img_size // 4)
        self.fc1 = nn.Linear(self.fc_size, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.fc_size)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AdvancedCNN(nn.Module):
    """进阶结构：增加卷积层、通道数，并引入 Dropout [cite: 61-65]"""
    def __init__(self, in_channels, img_size):
        super(AdvancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5) # 加入 Dropout 防止过拟合
        
        # 经过两次池化，尺寸缩小到 1/4
        self.fc_size = 128 * (img_size // 4) * (img_size // 4)
        self.fc1 = nn.Linear(self.fc_size, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x))) # 第一层池化
        x = self.pool(self.relu(self.conv3(x))) # 第二层池化
        x = x.view(-1, self.fc_size)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_and_evaluate(model, loaders, optimizer_name, lr, epochs=5):
    train_loader, val_loader, test_loader = loaders
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total

def run_advanced_experiments():
    print(f"正在使用 {device} 运行进阶实验")
    
    mnist_loaders = get_dataloaders('MNIST')
    cifar_loaders = get_dataloaders('CIFAR10')
    
    print("\n--- [进阶任务 1] 网络结构对比 (MNIST) ---")
    base_model = BaseCNN(mnist_loaders[3], mnist_loaders[4]).to(device)
    adv_model = AdvancedCNN(mnist_loaders[3], mnist_loaders[4]).to(device)
    
    acc_base = train_and_evaluate(base_model, mnist_loaders[:3], 'Adam', 0.001)
    acc_adv = train_and_evaluate(adv_model, mnist_loaders[:3], 'Adam', 0.001)
    print(f"基础网络 测试准确率: {acc_base:.2f}%")
    print(f"进阶网络 测试准确率: {acc_adv:.2f}%")

    print("\n--- [进阶任务 2] 优化器对比记录表 ---")
    print("Optimizer\tLearning Rate\tTest Accuracy")

    model_sgd = BaseCNN(mnist_loaders[3], mnist_loaders[4]).to(device)
    acc_sgd = train_and_evaluate(model_sgd, mnist_loaders[:3], 'SGD', 0.01) # SGD 通常用较大的学习率
    print(f"SGD\t\t0.01\t\t{acc_sgd:.2f}%")
    print(f"Adam\t\t0.001\t\t{acc_base:.2f}%") # 复用刚才的基础网络结果

    print("\n--- [进阶任务 3] MNIST 与 CIFAR-10 比较记录表 ---")
    adv_model_cifar = AdvancedCNN(cifar_loaders[3], cifar_loaders[4]).to(device)
    
    print("正在训练 CIFAR-10 模型 (耗时会稍长)...")
    acc_cifar = train_and_evaluate(adv_model_cifar, cifar_loaders[:3], 'Adam', 0.001, epochs=5)
    
    print("数据集\t图像类型\t类别数\t测试准确率\t难度")
    print(f"MNIST\t灰度手写数字\t10\t{acc_adv:.2f}%\t\t低")
    print(f"CIFAR-10\t彩色自然图像\t10\t{acc_cifar:.2f}%\t\t高")

if __name__ == "__main__":
    run_advanced_experiments()