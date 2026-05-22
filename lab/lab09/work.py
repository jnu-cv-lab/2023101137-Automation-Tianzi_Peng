import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def train_model(model, trainloader, testloader, criterion, optimizer, device, epochs=5):
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 记录 batch 的 loss 和准确率
            running_train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = 100 * correct_train / total_train
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = 100 * correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
    # 最后一次 validation accuracy 作为 test accuracy 输出
    test_acc = history['val_acc'][-1]
    
    return history, test_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs('./lab/lab09/output_images', exist_ok=True)
    
    trainloader, testloader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()

    print("--- 任务 2：优化器对比 (SGD, Momentum, Adam) ---")
    optimizers_dict = {
        'SGD': lambda m: optim.SGD(m.parameters(), lr=0.01),
        'SGD+Momentum': lambda m: optim.SGD(m.parameters(), lr=0.01, momentum=0.9),
        'Adam': lambda m: optim.Adam(m.parameters(), lr=0.01)
    }
    
    results_opt = {}
    for name, opt_func in optimizers_dict.items():
        print(f"\nTraining with {name}...")
        model = SimpleCNN().to(device)
        optimizer = opt_func(model)
        
        history, test_acc = train_model(model, trainloader, testloader, criterion, optimizer, device, epochs=5)
        results_opt[name] = {'history': history, 'test_acc': test_acc}
        
        print(f"[{name} Final Metrics] "
              f"Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {history['val_loss'][-1]:.4f} | "
              f"Train Acc: {history['train_acc'][-1]:.2f}% | "
              f"Val Acc: {history['val_acc'][-1]:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

    print("\n--- 任务 3：学习率对比 (Adam: 0.1, 0.01, 0.001) ---")
    lrs = [0.1, 0.01, 0.001]
    results_lr = {}
    best_model = None
    best_acc = 0

    for lr in lrs:
        print(f"Training Adam with lr={lr}...")
        model = SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        history, test_acc = train_model(model, trainloader, testloader, criterion, optimizer, device, epochs=5)
        results_lr[lr] = {'history': history, 'test_acc': test_acc}
        
        print(f"LR={lr} Test Accuracy: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model

    print(">>> 正在生成学习率对比曲线图片 (Loss & Accuracy)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    epochs_range = range(1, 6) # 3个 epoch
    colors = {0.1: 'red', 0.01: 'green', 0.001: 'blue'}
    
    for lr, res in results_lr.items():
        hist = res['history']

        ax1.plot(epochs_range, hist['train_loss'], color=colors[lr], linestyle='-', label=f'Train (lr={lr})')
        ax1.plot(epochs_range, hist['val_loss'], color=colors[lr], linestyle='--', alpha=0.7, label=f'Val (lr={lr})')
        
        ax2.plot(epochs_range, hist['train_acc'], color=colors[lr], linestyle='-', label=f'Train (lr={lr})')
        ax2.plot(epochs_range, hist['val_acc'], color=colors[lr], linestyle='--', alpha=0.7, label=f'Val (lr={lr})')

    ax1.set_title('Task 3: Loss Curves across Learning Rates')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Cross Entropy)')
    ax1.set_xticks(epochs_range)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.set_title('Task 3: Accuracy Curves across Learning Rates')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xticks(epochs_range)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('./lab/lab09/output_images/task3_lr_curves.png')
    plt.close()

    print("--- 任务 4：生成卷积核可视化图片 ---")
    kernels = best_model.conv1.weight.data.cpu().numpy()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7)) 
    for i, ax in enumerate(axes.flat):
        if i < 8:
            kernel = kernels[i, 0, :, :]
            im = ax.imshow(kernel, cmap='viridis')
            ax.set_title(f'Kernel {i+1}')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            for row in range(kernel.shape[0]):
                for col in range(kernel.shape[1]):
                    val = kernel[row, col]
                    text_color = "white" if val < kernel.mean() else "black"
                    ax.text(col, row, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10)
            ax.axis('off')
    plt.suptitle("Task 4: First Layer Convolutional Kernels (with Colorbar & Values)", fontsize=16)
    plt.tight_layout()
    plt.savefig('./lab/lab09/output_images/task4_kernels_with_legend.png')
    plt.close()

    print("--- 任务 5：生成 Feature map 图片 ---")
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    sample_image = images[0:1].to(device)
    
    best_model.eval()
    with torch.no_grad():
        feature_maps = best_model.relu1(best_model.conv1(sample_image))
    feature_maps = feature_maps.cpu().numpy()[0]
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < 8:
            ax.imshow(feature_maps[i, :, :], cmap='gray')
            ax.set_title(f'Feature Map {i+1}')
            ax.axis('off')
    plt.suptitle("Task 5: First Layer Feature Maps")
    plt.savefig('./lab/lab09/output_images/task5_feature_maps.png')
    plt.close()

    print("--- 任务 6 & 7：生成错误分类图片与混淆矩阵 ---")
    all_preds = []
    all_labels = []
    misclassified_imgs = []
    misclassified_labels = []
    misclassified_preds = []

    best_model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            outputs = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            predicted_cpu = predicted.cpu()
            all_preds.extend(predicted_cpu.numpy())
            all_labels.extend(labels.numpy())
            
            wrong_idx = (predicted_cpu != labels).nonzero(as_tuple=True)[0]
            for idx in wrong_idx:
                misclassified_imgs.append(images[idx].cpu().numpy().squeeze())
                misclassified_labels.append(labels[idx].item())
                misclassified_preds.append(predicted_cpu[idx].item())

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < min(8, len(misclassified_imgs)):
            ax.imshow(misclassified_imgs[i], cmap='gray')
            ax.set_title(f"True: {misclassified_labels[i]} | Pred: {misclassified_preds[i]}", color="red")
            ax.axis('off')
    plt.suptitle("Task 6: Misclassified Samples")
    plt.savefig('./lab/lab09/output_images/task6_misclassified.png')
    plt.close()

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Task 7: Confusion Matrix")
    plt.savefig('./lab/lab09/output_images/task7_confusion_matrix.png')
    plt.close()

if __name__ == '__main__':
    main()