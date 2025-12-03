import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def build_vit(num_classes=32, freeze_backbone=True):
    # تحميل ViT pre-trained
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    
    # Freeze backbone layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # استبدال classifier layer
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model
