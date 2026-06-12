import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 经典稳定版 API 导入
import mediapipe as mp 

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import math

# ==========================================
# 全局超参数与配置
# ==========================================
CONFIG = {
    'input_dim': 132,
    'target_frames': 30,
    'd_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'num_classes': 6,
    'dropout': 0.1,
    'batch_size': 32,
    'epochs': 30,
    'lr': 1e-3,
    'data_dir': './badminton_storke_video', 
    'output_dir': './output_images',
    'class_names': ['forehand_drive', 'forehand_lift', 'forehand_net_shot', 
                    'forehand_clear', 'backhand_drive', 'backhand_net_shot']
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 任务 1 & 6：视频预处理与骨架提取 (MediaPipe)
# ==========================================
def extract_skeleton_from_video(video_path, target_frames=30):
    """提取单个视频的骨架序列，并重采样到目标帧数"""
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_features = []
            for lm in landmarks:
                frame_features.extend([lm.x, lm.y, lm.z, lm.visibility])
            frames_data.append(frame_features)
            
    cap.release()
    pose.close()
    
    if len(frames_data) == 0:
        return None
        
    frames_data = np.array(frames_data) 
    
    # 线性插值重采样
    T_orig = frames_data.shape[0]
    indices = np.linspace(0, T_orig - 1, target_frames)
    resampled_data = np.zeros((target_frames, CONFIG['input_dim']))
    for i in range(CONFIG['input_dim']):
        resampled_data[:, i] = np.interp(indices, np.arange(T_orig), frames_data[:, i])
        
    # 去均值中心化
    resampled_data = resampled_data - np.mean(resampled_data, axis=0)
    
    return resampled_data

def prepare_dataset(data_dir):
    """遍历数据集文件夹，通过文件夹名称自动分配标签"""
    X, y = [], []
    if os.path.exists(data_dir):
        print(f"正在从 {data_dir} 提取视频骨架特征，这可能需要几分钟...")
        
        class_to_idx = {class_name: idx for idx, class_name in enumerate(CONFIG['class_names'])}
        
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            
            # 严格匹配文件夹名字，必须和 class_names 里面的一模一样
            if not os.path.isdir(class_dir) or class_name not in class_to_idx:
                continue
                
            label_idx = class_to_idx[class_name]
            print(f"  > 正在处理类别: {class_name} (标签索引: {label_idx})")
            
            for file in os.listdir(class_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(class_dir, file)
                    features = extract_skeleton_from_video(video_path, CONFIG['target_frames'])
                    if features is not None:
                        X.append(features)
                        y.append(label_idx)
    
    if len(X) == 0:
        print("未找到真实视频数据，请严格检查文件夹结构与命名。")
        return None, None, None, None
    else:
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        print(f"成功提取了 {len(X)} 个有效视频的骨架数据。")
        
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 任务 7：模型设计 (Skeleton Transformer)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SkeletonTransformer(nn.Module):
    def __init__(self, cfg):
        super(SkeletonTransformer, self).__init__()
        self.embedding = nn.Linear(cfg['input_dim'], cfg['d_model'])
        self.pos_encoder = PositionalEncoding(cfg['d_model'], cfg['dropout'], cfg['target_frames'])
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=cfg['d_model'], 
            nhead=cfg['nhead'], 
            dim_feedforward=cfg['dim_feedforward'], 
            dropout=cfg['dropout'],
            batch_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, cfg['num_layers'])
        
        self.classifier = nn.Sequential(
            nn.Linear(cfg['d_model'], 64),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(64, cfg['num_classes'])
        )

    def forward(self, x):
        x = self.embedding(x)                
        x = self.pos_encoder(x)              
        x = self.transformer_encoder(x)      
        x = x.mean(dim=1)                    
        logits = self.classifier(x)          
        return logits

# ==========================================
# 任务 8：训练与测试流水线
# ==========================================
def train_and_evaluate():
    X_train, X_test, y_train, y_test = prepare_dataset(CONFIG['data_dir'])
    
    if X_train is None:
        return None
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model = SkeletonTransformer(CONFIG).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    train_losses, test_accs = [], []
    
    print("\n--- 开始训练 Transformer Encoder ---")
    for epoch in range(CONFIG['epochs']):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        epoch_acc = accuracy_score(all_targets, all_preds) * 100
        test_accs.append(epoch_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] | Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.2f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, color='blue', label='Train Loss')
    ax1.set_title('Training Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CrossEntropy Loss')
    ax1.grid(True)
    
    ax2.plot(test_accs, color='green', label='Test Accuracy')
    ax2.set_title('Test Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    plt.savefig(os.path.join(CONFIG['output_dir'], 'training_curves.png'))
    plt.close()
    
    print("\n--- 最终分类报告 ---")
    print(classification_report(all_targets, all_preds, target_names=CONFIG['class_names']))
    
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(CONFIG['num_classes']))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Confusion Matrix of Badminton Strokes")
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'))
    plt.close()
    
    return model

# ==========================================
# 任务 9：单样本推理 (Inference)
# ==========================================
def inference_single_video(model, video_path):
    print(f"\n--- 执行单样本推理: {video_path} ---")
    if not os.path.exists(video_path):
        print("未找到指定视频，跳过推理环节。")
        return
        
    model.eval()
    features = extract_skeleton_from_video(video_path, CONFIG['target_frames'])
    if features is None:
        print("提取骨架失败。")
        return
        
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
    pred_class_idx = np.argmax(probs)
    confidence = probs[pred_class_idx]
    
    print("推理结果:")
    print(f"Predicted class: {CONFIG['class_names'][pred_class_idx]}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == '__main__':
    trained_model = train_and_evaluate()
    
    if trained_model is not None:
        demo_path = os.path.join(CONFIG['data_dir'], 'demo_video.mp4')
        inference_single_video(trained_model, demo_path)
        print("\n实验运行完毕，曲线和混淆矩阵已保存至 output_images 目录。")