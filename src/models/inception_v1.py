import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

def build_inception_v1(num_classes=32, freeze_backbone=True, pretrained=True):
    # تحميل الأوزان المسبقة إذا كان المطلوب
    weights = GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = googlenet(weights=weights, aux_logits=True)  # aux_logits=True لتوافق الأوزان

    # Freeze backbone layers لعمل transfer learning
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # استبدال آخر طبقة fully connected بعدد الفئات الجديدة
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
