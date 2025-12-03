import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def build_resnet(num_classes=32, freeze_backbone=True):
    # تحميل ResNet50 مع weights مسبقة من ImageNet
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    
    # Freeze backbone layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # استبدال آخر طبقة fully connected لتتناسب مع عدد الفئات
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
