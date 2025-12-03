import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# ---- Add project root to sys.path ----
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# ---------------------- IMPORT MODELS ----------------------
from models.vgg19 import build_vgg19
from models.resnet import build_resnet
from models.inception_v1 import build_inception_v1
from models.vit import build_vit
from models.vgg19_pretrained import build_vgg19_pretrained

# ---------------------- CONFIG ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 32  # Flavia dataset
batch_size = 32
data_dir = "data/flavia"
checkpoint_epoch = 15

# ---------------------- DATA ----------------------
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------- MODELS & CHECKPOINTS ----------------------
models_info = {
    "VGG19": "outputs/vgg19/checkpoints/vgg19_epoch_15.pth",
    "ResNet": "outputs/resnet/checkpoints/resnet_epoch_15.pth",
    "Inception": "outputs/inception_v1/checkpoints/inception_v1_epoch_15.pth",
    "ViT": "outputs/vit/checkpoints/vit_epoch_15.pth",
    "vgg19_pretrained":"outputs/vgg19_pretrained/checkpoints/vgg19_pretrained_epoch_15.pth"
}

# ---------------------- EVALUATION FUNCTION ----------------------
def evaluate_model(model, checkpoint_path, model_name):
    print(f"\n📥 Loading checkpoint for {model_name} from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ---- Metrics ----
    acc = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro")
    recall_macro = recall_score(all_labels, all_preds, average="macro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    # ---- Save directory ----
    save_dir = os.path.join(os.path.dirname(checkpoint_path), "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # ---- Metrics per class bar chart ----
    x = np.arange(num_classes)
    width = 0.25
    plt.figure(figsize=(16, 7))
    plt.bar(x - width, precision_per_class, width, label="Precision")
    plt.bar(x, recall_per_class, width, label="Recall")
    plt.bar(x + width, f1_per_class, width, label="F1-score")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title(f"{model_name} Metrics per Class")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, f"{model_name}_metrics_per_class.png"))
    plt.close()

    # ---- ROC curves per class ----
    one_hot = label_binarize(all_labels, classes=list(range(num_classes)))
    plt.figure(figsize=(16, 12))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(one_hot[:, i], all_probs[:, i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc_score:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curves")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"{model_name}_roc_curves.png"))
    plt.close()

    # ---- AUC bar chart ----
    auc_scores = [auc(roc_curve(one_hot[:, i], all_probs[:, i])[0], 
                      roc_curve(one_hot[:, i], all_probs[:, i])[1]) for i in range(num_classes)]
    plt.figure(figsize=(14, 7))
    plt.bar(range(num_classes), auc_scores)
    plt.xlabel("Class")
    plt.ylabel("AUC Score")
    plt.title(f"{model_name} AUC per Class")
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, f"{model_name}_auc_bar_chart.png"))
    plt.close()

    print(f"✔ {model_name} evaluation completed and saved to {save_dir}")
    print(f"====== 📊 Evaluation Results {model_name} ======")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall: {recall_macro:.4f}")
    print(f"F1-score: {f1_macro:.4f}")

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": precision_macro,
        "Recall": recall_macro,
        "F1": f1_macro
    }
    
# ---------------------- EVALUATE ALL MODELS ----------------------
results = []
for model_name, ckpt_path in models_info.items():
    if model_name == "VGG19":
        model = build_vgg19(num_classes)
    elif model_name == "ResNet":
        model = build_resnet(num_classes)
    elif model_name == "Inception":
        model = build_inception_v1(num_classes)
    elif model_name == "ViT":
        model = build_vit(num_classes)
    elif model_name == "vgg19_pretrained":
        model = build_vgg19_pretrained(num_classes)

    res = evaluate_model(model, ckpt_path, model_name)
    results.append(res)

# ---------------------- Comparison Plot ----------------------
plt.figure(figsize=(10, 6))
x = np.arange(len(results))
width = 0.2
accs = [r["Accuracy"] for r in results]
precisions = [r["Precision"] for r in results]
recalls = [r["Recall"] for r in results]
f1s = [r["F1"] for r in results]

plt.bar(x - width*1.5, accs, width, label="Accuracy")
plt.bar(x - width/2, precisions, width, label="Precision")
plt.bar(x + width/2, recalls, width, label="Recall")
plt.bar(x + width*1.5, f1s, width, label="F1-score")

plt.xticks(x, [r["Model"] for r in results])
plt.ylim(0, 1.05)
plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Comparison of Models on Test Set (Checkpoint 15)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
os.makedirs("evaluation", exist_ok=True)
plt.savefig("evaluation/comparison_all_models.png")
plt.close()

print("\n✔ Comparison chart saved to evaluation/comparison_all_models.png")
