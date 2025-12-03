import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from models.resnet import build_resnet
from tqdm import tqdm

# ------------------------
# Argument Parser
# ------------------------
parser = argparse.ArgumentParser(description="Train ResNet on Flavia dataset")
parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset folder")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save outputs and checkpoints")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr

# ------------------------
# Hyperparameters
# ------------------------
num_classes = 32  # Flavia dataset
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Datasets & DataLoaders
# ------------------------
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
transform_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_path = os.path.join(data_dir, "train")
val_path = os.path.join(data_dir, "val")

print("DEBUG: train_path =", train_path)
print("DEBUG: val_path =", val_path)

train_dataset = datasets.ImageFolder(train_path, transform_train)
val_dataset = datasets.ImageFolder(val_path, transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------------
# Model, Loss, Optimizer
# ------------------------
model = build_resnet(num_classes=num_classes, freeze_backbone=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=lr)  # فقط layer الأخيرة قابلة للتعلم

# ------------------------
# Training Loop with tqdm
# ------------------------
best_val_loss = float('inf')
counter = 0

for epoch in range(epochs):
    # --- Training ---
    model.train()
    running_loss, correct, total = 0, 0, 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train", unit="batch")
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        train_loader_tqdm.set_postfix(loss=running_loss/total, acc=correct/total)

    train_loss = running_loss / total
    train_acc = correct / total

    # --- Validation ---
    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Val", unit="batch")
    with torch.no_grad():
        for images, labels in val_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct_val += predicted.eq(labels).sum().item()
            total_val += labels.size(0)

            val_loader_tqdm.set_postfix(val_loss=val_loss/total_val, val_acc=correct_val/total_val)

    val_loss /= total_val
    val_acc = correct_val / total_val

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # --- Save checkpoint for every epoch ---
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"resnet_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

