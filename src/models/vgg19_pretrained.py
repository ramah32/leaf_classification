import torch
import torch.nn as nn
from torchvision import models

def build_vgg19_pretrained(num_classes=32, freeze_features=True):
    # تحميل نموذج VGG19 بوزن pretrained ImageNet
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

    # تجميد الطبقات إن كنتِ تريدين transfer learning فقط
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # تعديل طبقة الإخراج فقط لتناسب عدد الأصناف لديك
    model.classifier[6] = nn.Linear(4096, num_classes)

    return model
