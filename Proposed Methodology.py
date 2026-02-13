#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## ====================================================================
## Step 1: Install and Import
## ====================================================================
# !pip install -q opendatasets einops scikit-learn seaborn

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
import opendatasets as od

## ====================================================================
## Step 2: Download Dataset
## ====================================================================
dataset_url = 'https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset'
data_dir = './breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/'

if not os.path.exists(data_dir):
    od.download(dataset_url)

## ====================================================================
## Step 3: Prepare Stratified 70/15/15 Split
## ====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.RandomAffine(10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dataset_full = datasets.ImageFolder(data_dir)
class_names = dataset_full.classes
targets = dataset_full.targets
indices = list(range(len(targets)))
targets_array = np.array(targets)

train_idx, temp_idx = train_test_split(
    indices, stratify=targets, test_size=0.3, random_state=42)

val_idx, test_idx = train_test_split(
    temp_idx, stratify=targets_array[temp_idx], test_size=0.5, random_state=42)

train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val_test'])
test_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val_test'])

train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=32, shuffle=True)
val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=32, shuffle=False)
test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=32, shuffle=False)

## ====================================================================
## Step 4: Model Definition
## ====================================================================
class CnnGruViT(nn.Module):
    def __init__(self, num_classes, cnn_output_dim=512, gru_hidden_dim=512, vit_heads=8, vit_depth=4, dropout=0.5):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for name, param in resnet.named_parameters():
            if 'layer2' in name or 'layer3' in name or 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gru = nn.GRU(input_size=cnn_output_dim, hidden_size=gru_hidden_dim, num_layers=2,
                          batch_first=True, dropout=dropout, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=gru_hidden_dim*2, nhead=vit_heads,
                                                   dim_feedforward=gru_hidden_dim*4, dropout=dropout, batch_first=True)
        self.vit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=vit_depth)
        self.classifier = nn.Linear(gru_hidden_dim*2, num_classes)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, gru_hidden_dim*2))

    def forward(self, x):
        features = self.cnn_backbone(x)
        features = rearrange(features, 'b c h w -> b (h w) c')
        output, _ = self.gru(features)
        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
        output = torch.cat((cls_tokens, output), dim=1)
        vit_output = self.vit_encoder(output)
        return self.classifier(vit_output[:, 0])

## ====================================================================
## Step 5: Training and Evaluation Functions
## ====================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        _, pred = torch.max(y_hat, 1)
        correct += torch.sum(pred == y)
        total += y.size(0)
    return total_loss / total, correct.double() / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            total_loss += loss.item() * x.size(0)
            _, pred = torch.max(y_hat, 1)
            correct += torch.sum(pred == y)
            total += y.size(0)
            y_true += y.cpu().tolist()
            y_pred += pred.cpu().tolist()
            y_score += torch.softmax(y_hat, dim=1).cpu().tolist()
    return total_loss / total, correct.double() / total, y_true, y_pred, y_score

## ====================================================================
## Step 6: Training Loop
## ====================================================================
model = CnnGruViT(num_classes=len(class_names)).to(device)

class_counts = np.bincount(targets_array[train_idx])
class_weights = 1. / class_counts
weights = torch.tensor([class_weights[i] for i in range(len(class_names))], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

train_acc_hist, val_acc_hist = [], []
train_loss_hist, val_loss_hist = [], []

for epoch in range(25):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)
    scheduler.step()

    train_loss_hist.append(float(train_loss))
    val_loss_hist.append(float(val_loss))
    train_acc_hist.append(train_acc.item())
    val_acc_hist.append(val_acc.item())


    print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

## ====================================================================
## Step 7: Test Evaluation + Visualization
## ====================================================================
_, test_acc, y_true, y_pred, y_score = evaluate(model, test_loader, criterion)
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

# Plot accuracy and loss
epochs = range(1, 26)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc_hist, label='Train Acc')
plt.plot(epochs, val_acc_hist, label='Val Acc')
plt.title('Accuracy over Epochs - CNN-GRU-ViT')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss_hist, label='Train Loss')
plt.plot(epochs, val_loss_hist, label='Val Loss')
plt.title('Loss over Epochs - CNN-GRU-ViT')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix- CNN-GRU-ViT')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ROC AUC
y_true_bin = np.eye(len(class_names))[y_true]
y_score = np.array(y_score)
plt.figure(figsize=(8, 6))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves- CNN-GRU-ViT')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


## ====================================================================
## Step 1: Install and Import
## ====================================================================
# !pip install -q opendatasets einops scikit-learn seaborn

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
import opendatasets as od

## ====================================================================
## Step 2: Download Dataset
## ====================================================================
dataset_url = 'https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset'
data_dir = './breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/'

if not os.path.exists(data_dir):
    od.download(dataset_url)

## ====================================================================
## Step 3: Prepare Dataset (70/15/15 Split)
## ====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.RandomAffine(10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dataset_full = datasets.ImageFolder(data_dir)
class_names = dataset_full.classes
targets = dataset_full.targets
indices = list(range(len(targets)))
targets_array = np.array(targets)

train_idx, temp_idx = train_test_split(indices, stratify=targets, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, stratify=targets_array[temp_idx], test_size=0.5, random_state=42)

train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val_test'])
test_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val_test'])

train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=32, shuffle=True)
val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=32, shuffle=False)
test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=32, shuffle=False)

## ====================================================================
## Step 4: Define Models
## ====================================================================
class CnnGruViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for name, param in resnet.named_parameters():
            if 'layer2' in name or 'layer3' in name or 'layer4' in name:
                param.requires_grad = True
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gru = nn.GRU(512, 512, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8, dim_feedforward=2048, dropout=0.5, batch_first=True)
        self.vit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(1024, num_classes)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1024))

    def forward(self, x):
        features = self.cnn_backbone(x)
        features = rearrange(features, 'b c h w -> b (h w) c')
        output, _ = self.gru(features)
        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
        output = torch.cat((cls_tokens, output), dim=1)
        vit_output = self.vit_encoder(output)
        return self.classifier(vit_output[:, 0])

def get_mobilenet(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = True
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

## ====================================================================
## Step 5: Training Utilities
## ====================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_total, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_total += loss.item() * x.size(0)
        _, pred = torch.max(out, 1)
        correct += torch.sum(pred == y)
        total += y.size(0)
    return float(loss_total) / total, correct.double().item() / total

def evaluate_model(model, loader, criterion):
    model.eval()
    loss_total, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_total += loss.item() * x.size(0)
            _, pred = torch.max(out, 1)
            correct += torch.sum(pred == y)
            total += y.size(0)
    return float(loss_total) / total, correct.double().item() / total

## ====================================================================
## Step 6: Train Both Models
## ====================================================================
model1 = CnnGruViT(num_classes=len(class_names)).to(device)
model2 = get_mobilenet(num_classes=len(class_names)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=1e-4)
optimizer2 = optim.Adam(model2.parameters(), lr=1e-4)

train_acc1, val_acc1, train_loss1, val_loss1 = [], [], [], []
train_acc2, val_acc2, train_loss2, val_loss2 = [], [], [], []

for epoch in range(50):  # Shortened for demo
    tl1, ta1 = train_one_epoch(model1, train_loader, criterion, optimizer1)
    vl1, va1 = evaluate_model(model1, val_loader, criterion)

    tl2, ta2 = train_one_epoch(model2, train_loader, criterion, optimizer2)
    vl2, va2 = evaluate_model(model2, val_loader, criterion)

    train_loss1.append(tl1); val_loss1.append(vl1); train_acc1.append(ta1); val_acc1.append(va1)
    train_loss2.append(tl2); val_loss2.append(vl2); train_acc2.append(ta2); val_acc2.append(va2)

    print(f"Epoch {epoch+1:02d} | CNN-GRU-ViT Acc: {va1:.4f} | MobileNet Acc: {va2:.4f}")

## ====================================================================
## Step 7: Evaluate Ensemble
## ====================================================================
def evaluate_ensemble(model1, model2, loader):
    model1.eval()
    model2.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p1 = torch.softmax(model1(x), dim=1)
            p2 = torch.softmax(model2(x), dim=1)
            ensemble_output = (p1 + p2) / 2
            _, preds = torch.max(ensemble_output, 1)
            y_true += y.cpu().tolist()
            y_pred += preds.cpu().tolist()
            y_score += ensemble_output.cpu().tolist()
    return y_true, y_pred, y_score

y_true, y_pred, y_score = evaluate_ensemble(model1, model2, test_loader)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - CNN-GRU-ViT with MobileNetV2')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# ROC AUC
y_true_bin = np.eye(len(class_names))[y_true]
y_score = np.array(y_score)
plt.figure(figsize=(8, 6))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves - CNN-GRU-ViT with MobileNetV2')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Training Curve
epochs = range(1, len(train_acc1)+1)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc1, label='Train Acc CNN-GRU-ViT')
plt.plot(epochs, val_acc1, label='Val Acc CNN-GRU-ViT')
plt.plot(epochs, train_acc2, label='Train Acc MobileNet')
plt.plot(epochs, val_acc2, label='Val Acc MobileNet')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss1, label='Train Loss CNN-GRU-ViT')
plt.plot(epochs, val_loss1, label='Val Loss CNN-GRU-ViT')
plt.plot(epochs, train_loss2, label='Train Loss MobileNet')
plt.plot(epochs, val_loss2, label='Val Loss MobileNet')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


## ====================================================================
## Step 1: Install and Import
## ====================================================================
# !pip install -q opendatasets einops scikit-learn seaborn

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
import opendatasets as od

## ====================================================================
## Step 2: Dataset and Preprocessing
## ====================================================================
dataset_url = 'https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset'
data_dir = './breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/'

if not os.path.exists(data_dir):
    od.download(dataset_url)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resize to 64x64 and flatten for LSTM input (sequence length = 64, feature dim = 64 * 3)
IMG_SIZE = 64

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]),
}

dataset = datasets.ImageFolder(data_dir)
class_names = dataset.classes
targets = dataset.targets
indices = list(range(len(targets)))
targets_array = np.array(targets)

train_idx, temp_idx = train_test_split(indices, stratify=targets, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, stratify=targets_array[temp_idx], test_size=0.5, random_state=42)

train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val_test'])
test_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val_test'])

train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=32, shuffle=True)
val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=32, shuffle=False)
test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=32, shuffle=False)

## ====================================================================
## Step 3: LSTM-Only Model
## ====================================================================
class LSTMOnlyClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (B, 3, 64, 64) -> (B, 64, 64*3)
        x = x.view(x.size(0), IMG_SIZE, -1)
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1])

## ====================================================================
## Step 4: Training and Evaluation
## ====================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        _, pred = torch.max(out, 1)
        correct += torch.sum(pred == y)
        total += y.size(0)
    return float(total_loss) / total, correct.double().item() / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            _, pred = torch.max(out, 1)
            correct += torch.sum(pred == y)
            total += y.size(0)
            y_true += y.cpu().tolist()
            y_pred += pred.cpu().tolist()
            y_score += torch.softmax(out, dim=1).cpu().tolist()
    return float(total_loss) / total, correct.double().item() / total, y_true, y_pred, y_score

## ====================================================================
## Step 5: Train the Model
## ====================================================================
model = LSTMOnlyClassifier(input_size=IMG_SIZE * 3, hidden_size=128, num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_acc, val_acc, train_loss, val_loss = [], [], [], []

for epoch in range(25):
    tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
    vl, va, _, _, _ = evaluate(model, val_loader, criterion)

    train_loss.append(tl)
    val_loss.append(vl)
    train_acc.append(ta)
    val_acc.append(va)

    print(f"Epoch {epoch+1:02d} | Train Acc: {ta:.4f} | Val Acc: {va:.4f}")

## ====================================================================
## Step 6: Final Evaluation & Visualization
## ====================================================================
_, _, y_true, y_pred, y_score = evaluate(model, test_loader, criterion)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix -LSTM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

y_true_bin = np.eye(len(class_names))[y_true]
y_score = np.array(y_score)
plt.figure(figsize=(8, 6))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves - LSTM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Accuracy and Loss Plots
epochs = range(1, len(train_acc)+1)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Train Acc')
plt.plot(epochs, val_acc, label='Val Acc')
plt.title('Accuracy Over Epochs - LSTM')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# ===============================================================
# ResNet-50 for Breast Ultrasound Image Classification
# ===============================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================================
# Dataset Setup
# ===============================================================
IMG_SIZE = 224
data_dir = './breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/'  # download manually or via opendatasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dataset = datasets.ImageFolder(data_dir)
class_names = dataset.classes
targets = dataset.targets
indices = list(range(len(targets)))
targets_array = np.array(targets)

train_idx, temp_idx = train_test_split(indices, stratify=targets, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, stratify=targets_array[temp_idx], test_size=0.5, random_state=42)

train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val_test'])
test_dataset = datasets.ImageFolder(data_dir, transform=data_transforms['val_test'])

train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=32, shuffle=True)
val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=32, shuffle=False)
test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=32, shuffle=False)

# ===============================================================
# Model Setup
# ===============================================================
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ===============================================================
# Training Functions
# ===============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        _, pred = torch.max(out, 1)
        correct += torch.sum(pred == y)
        total += y.size(0)
    return total_loss / total, correct.double().item() / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            _, pred = torch.max(out, 1)
            correct += torch.sum(pred == y)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            y_score.extend(torch.softmax(out, dim=1).cpu().tolist())
            total += y.size(0)
    return total_loss / total, correct.double().item() / total, y_true, y_pred, y_score

# ===============================================================
# Training Loop
# ===============================================================
train_loss, val_loss, train_acc, val_acc = [], [], [], []
for epoch in range(25):  # adjust epochs as needed
    tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
    vl, va, _, _, _ = evaluate(model, val_loader, criterion)
    train_loss.append(tl)
    val_loss.append(vl)
    train_acc.append(ta)
    val_acc.append(va)
    print(f"Epoch {epoch+1:02d}: Train Acc={ta:.4f}, Val Acc={va:.4f}")

# ===============================================================
# Final Evaluation and Plots
# ===============================================================
_, _, y_true, y_pred, y_score = evaluate(model, test_loader, criterion)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - ResNet-50')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ROC Curve
y_true_bin = np.eye(len(class_names))[y_true]
y_score = np.array(y_score)

plt.figure(figsize=(8, 6))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves - ResNet-50")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Accuracy and Loss Plot
epochs = range(1, len(train_acc)+1)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Train Acc')
plt.plot(epochs, val_acc, label='Val Acc')
plt.title('Accuracy Over Epochs - ResNet-50')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('Loss Over Epochs - ResNet-50')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# Accuracy and Loss Plot
epochs = range(1, len(train_acc)+1)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Train Acc')
plt.plot(epochs, val_acc, label='Val Acc')
plt.title('Accuracy Over Epochs - CNN-GRU-ViT + ResNet-50')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('Loss Over Epochs - CNN-GRU-ViT + ResNet-50')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# Training Curve
epochs = range(1, len(train_acc1)+1)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc1, label='Train Acc CNN-GRU-ViT')
plt.plot(epochs, val_acc1, label='Val Acc CNN-GRU-ViT')
plt.plot(epochs, train_acc2, label='Train Acc ResNet-50')
plt.plot(epochs, val_acc2, label='Val Acc ResNet-50')
plt.title('Accuracy over Epochs - CNN-GRU-ViT + ResNet-50')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss1, label='Train Loss CNN-GRU-ViT')
plt.plot(epochs, val_loss1, label='Val Loss CNN-GRU-ViT')
plt.plot(epochs, train_loss2, label='Train Loss ResNet-50')
plt.plot(epochs, val_loss2, label='Val Loss ResNet-50')
plt.title('Loss over Epochs - CNN-GRU-ViT + ResNet-50')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# ===============================================================
# GRU-Only Model for Breast Ultrasound Image Classification
# ===============================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================================
# Dataset Setup
# ===============================================================
IMG_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 25
data_dir = './breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/'  # make sure it is downloaded

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
}

dataset = datasets.ImageFolder(data_dir)
class_names = dataset.classes
targets = dataset.targets
indices = list(range(len(targets)))
targets_array = np.array(targets)

train_idx, temp_idx = train_test_split(indices, stratify=targets, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, stratify=targets_array[temp_idx], test_size=0.5, random_state=42)

train_dataset = datasets.ImageFolder(data_dir, transform=transform['train'])
val_dataset = datasets.ImageFolder(data_dir, transform=transform['val_test'])
test_dataset = datasets.ImageFolder(data_dir, transform=transform['val_test'])

train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

# ===============================================================
# GRU-Only Model Definition
# ===============================================================
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Input shape: (B, 3, 64, 64) -> Reshape to (B, 64, 3*64) = (B, Seq, Features)
        x = x.view(x.size(0), IMG_SIZE, -1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

# ===============================================================
# Training & Evaluation Functions
# ===============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        _, preds = torch.max(out, 1)
        correct += torch.sum(preds == y)
        total += y.size(0)
    return total_loss / total, correct.double().item() / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            _, preds = torch.max(out, 1)
            correct += torch.sum(preds == y)
            y_true += y.cpu().tolist()
            y_pred += preds.cpu().tolist()
            y_score += torch.softmax(out, dim=1).cpu().tolist()
            total += y.size(0)
    return total_loss / total, correct.double().item() / total, y_true, y_pred, y_score

# ===============================================================
# Initialize and Train the Model
# ===============================================================
model = GRUClassifier(input_size=IMG_SIZE * 3, hidden_size=128, num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_acc, val_acc, train_loss, val_loss = [], [], [], []

for epoch in range(NUM_EPOCHS):
    tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
    vl, va, _, _, _ = evaluate(model, val_loader, criterion)
    train_loss.append(tl)
    val_loss.append(vl)
    train_acc.append(ta)
    val_acc.append(va)
    print(f"Epoch {epoch+1:02d} | Train Acc: {ta:.4f} | Val Acc: {va:.4f}")

# ===============================================================
# Final Evaluation
# ===============================================================
_, _, y_true, y_pred, y_score = evaluate(model, test_loader, criterion)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - GRU')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# ROC AUC
y_true_bin = np.eye(len(class_names))[y_true]
y_score = np.array(y_score)

plt.figure(figsize=(8, 6))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves - GRU")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Accuracy and Loss Plot
epochs = range(1, len(train_acc)+1)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Train Acc')
plt.plot(epochs, val_acc, label='Val Acc')
plt.title('Accuracy Over Epochs -GRU')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('Loss Over Epochs -GRU')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# ===============================================================
# CNN-Only Model for Breast Ultrasound Image Classification
# ===============================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================================
# Dataset Setup
# ===============================================================
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 25
data_dir = './breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/'  # ensure this is available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.ToTensor(),
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
}

dataset = datasets.ImageFolder(data_dir)
class_names = dataset.classes
targets = dataset.targets
indices = list(range(len(targets)))
targets_array = np.array(targets)

train_idx, temp_idx = train_test_split(indices, stratify=targets, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, stratify=targets_array[temp_idx], test_size=0.5, random_state=42)

train_dataset = datasets.ImageFolder(data_dir, transform=transform['train'])
val_dataset = datasets.ImageFolder(data_dir, transform=transform['val_test'])
test_dataset = datasets.ImageFolder(data_dir, transform=transform['val_test'])

train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

# ===============================================================
# Simple CNN Model
# ===============================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ===============================================================
# Training and Evaluation Functions
# ===============================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        _, pred = torch.max(out, 1)
        correct += torch.sum(pred == y)
        total += y.size(0)
    return total_loss / total, correct.double().item() / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            _, pred = torch.max(out, 1)
            correct += torch.sum(pred == y)
            y_true += y.cpu().tolist()
            y_pred += pred.cpu().tolist()
            y_score += torch.softmax(out, dim=1).cpu().tolist()
            total += y.size(0)
    return total_loss / total, correct.double().item() / total, y_true, y_pred, y_score

# ===============================================================
# Initialize and Train the Model
# ===============================================================
model = SimpleCNN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_acc, val_acc, train_loss, val_loss = [], [], [], []

for epoch in range(NUM_EPOCHS):
    tl, ta = train_one_epoch(model, train_loader, criterion, optimizer)
    vl, va, _, _, _ = evaluate(model, val_loader, criterion)
    train_loss.append(tl)
    val_loss.append(vl)
    train_acc.append(ta)
    val_acc.append(va)
    print(f"Epoch {epoch+1:02d} | Train Acc: {ta:.4f} | Val Acc: {va:.4f}")

# ===============================================================
# Final Evaluation
# ===============================================================
_, _, y_true, y_pred, y_score = evaluate(model, test_loader, criterion)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - CNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ROC AUC
y_true_bin = np.eye(len(class_names))[y_true]
y_score = np.array(y_score)

plt.figure(figsize=(8, 6))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves - CNN")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Accuracy and Loss Plot
epochs = range(1, len(train_acc)+1)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Train Acc')
plt.plot(epochs, val_acc, label='Val Acc')
plt.title('Accuracy Over Epochs - CNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('Loss Over Epochs - CNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from einops import rearrange
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import opendatasets as od

# =================== Download Dataset ===================
od.download("https://www.kaggle.com/datasets/salmanbnr/breast-cancer-ultrasound-images")

base_dir = "./breast-cancer-ultrasound-images"
data_dir = None
for dirpath, dirnames, _ in os.walk(base_dir):
    if set(["benign", "malignant", "normal"]).issubset(set(dirnames)):
        data_dir = dirpath
        break

assert data_dir is not None, "Dataset class folders not found."
print(f"Dataset path: {data_dir}")

# =================== Config ===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

transform = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
}

# =================== Load Dataset ===================
dataset = datasets.ImageFolder(root=data_dir, transform=transform['train'])
class_names = dataset.classes
num_classes = len(class_names)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
val_set.dataset.transform = transform['val_test']
test_set.dataset.transform = transform['val_test']

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# =================== Model Definitions ===================
class CnnGruViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.gru = nn.GRU(512, 512, num_layers=2, batch_first=True, bidirectional=True)
        encoder = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1024))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x, _ = self.gru(x)
        cls = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.fc(x[:, 0])

def get_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# =================== Training Utilities ===================
def train_and_validate(model, train_loader, val_loader, criterion, optimizer):
    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    for epoch in range(EPOCHS):
        model.train()
        correct, total, t_loss = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc.append(correct / total)
        train_loss.append(t_loss / total)

        model.eval()
        correct, total, v_loss = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc.append(correct / total)
        val_loss.append(v_loss / total)
        print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc[-1]:.4f} | Val Acc: {val_acc[-1]:.4f}")
    return train_acc, val_acc, train_loss, val_loss

# =================== Initialize Models ===================
model1 = CnnGruViT(num_classes).to(device)
model2 = get_resnet50(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
opt1 = optim.Adam(model1.parameters(), lr=1e-4)
opt2 = optim.Adam(model2.parameters(), lr=1e-4)

print("Training CNN-GRU-ViT")
train_acc1, val_acc1, train_loss1, val_loss1 = train_and_validate(model1, train_loader, val_loader, criterion, opt1)
print("Training ResNet-50")
train_acc2, val_acc2, train_loss2, val_loss2 = train_and_validate(model2, train_loader, val_loader, criterion, opt2)

# =================== Ensemble Evaluation ===================
def ensemble_predict(m1, m2, loader):
    m1.eval()
    m2.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p1 = torch.softmax(m1(x), dim=1)
            p2 = torch.softmax(m2(x), dim=1)
            avg = (p1 + p2) / 2
            y_true.extend(y.tolist())
            y_pred.extend(avg.argmax(1).cpu().tolist())
            y_score.extend(avg.cpu().tolist())
    return y_true, y_pred, y_score

y_true, y_pred, y_score = ensemble_predict(model1, model2, test_loader)

# =================== Classification Report ===================
print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# =================== Confusion Matrix ===================
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - CNN-GRU-ViT + ResNet-50 (Second Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =================== ROC AUC Curve ===================
y_true_bin = np.eye(num_classes)[y_true]
y_score = np.array(y_score)

plt.figure(figsize=(8, 6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - CNN-GRU-ViT + ResNet-50 (Second Dataset)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =================== Accuracy and Loss Plots ===================
epochs = range(1, EPOCHS + 1)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc1, label='CNN-GRU-ViT Train Acc')
plt.plot(epochs, val_acc1, label='CNN-GRU-ViT Val Acc')
plt.plot(epochs, train_acc2, label='ResNet-50 Train Acc')
plt.plot(epochs, val_acc2, label='ResNet-50 Val Acc')
plt.title("Training and Validation Accuracy - CNN-GRU-ViT + ResNet-50 (Second Dataset)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss1, label='CNN-GRU-ViT Train Loss')
plt.plot(epochs, val_loss1, label='CNN-GRU-ViT Val Loss')
plt.plot(epochs, train_loss2, label='ResNet-50 Train Loss')
plt.plot(epochs, val_loss2, label='ResNet-50 Val Loss')
plt.title("Training and Validation Loss - CNN-GRU-ViT + ResNet-50 (Second Dataset)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

