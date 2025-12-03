import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.inception_v1 import build_inception_v1
from tqdm import tqdm
import argparse

# ------------------------
# Argument parser
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset folder')
parser.add_argument('--output_dir', type=str, required=True, help='Path to save outputs/checkpoints')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_classes', type=int, default=32)
parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained weights')
args = parser.parse_args()

# ------------------------
# Paths and device
# ------------------------
train_path = os.path.join(args.data_dir, "train")
val_path = os.path.join(args.data_dir, "val")

print("DEBUG: train_path =", train_path)
print("DEBUG: val_path =", val_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Datasets & Dataloaders
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

train_dataset = datasets.ImageFolder(train_path, transform_train)
val_dataset = datasets.ImageFolder(val_path, transform_val)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# ------------------------
# Model
# ------------------------
model = build_inception_v1(num_classes=args.num_classes, 
                           freeze_backbone=True, 
                           pretrained=args.use_pretrained).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

# ------------------------
# Training Loop with tqdm
# ------------------------
best_val_loss = float('inf')
patience = 5
counter = 0

# ------------------------
# Training Loop with tqdm
# ------------------------
for epoch in range(args.epochs):
    model.train()
    running_loss, correct, total = 0,0,0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Train", unit="batch")
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs  # تجاهل auxiliary outputs
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        train_loader_tqdm.set_postfix(loss=running_loss/total, acc=correct/total)
    
    train_loss = running_loss/total
    train_acc = correct/total

    # Validation
    model.eval()
    val_loss, correct_val, total_val = 0,0,0

    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Val", unit="batch")
    with torch.no_grad():
        for images, labels in val_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(outputs, labels)
            val_loss += loss.item()*images.size(0)
            _, predicted = outputs.max(1)
            correct_val += predicted.eq(labels).sum().item()
            total_val += labels.size(0)

            val_loader_tqdm.set_postfix(val_loss=val_loss/total_val, val_acc=correct_val/total_val)
    
    val_loss /= total_val
    val_acc = correct_val/total_val

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save checkpoint for every epoch
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "checkpoints", f"inception_v1_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


