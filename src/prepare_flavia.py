# scripts/prepare_flavia.py
import os
import shutil
import random
from pathlib import Path
from math import floor
from PIL import Image
import torch
from torchvision import transforms

random.seed(42)

# ======= ضبط المسارات =======
RAW_DIR = Path("data/Leaves")   # بعد فك الضغط
OUT_DIR = Path("data/flavia")   # ستكون البنية هنا
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======= Data Preprocessing Config =======
IMG_SIZE = (224, 224)

# ======= خريطة الأصناف + نطاق أرقام الملفات (مأخوذة من صفحة Flavia) =======
# كل مدخل: (label_id, class_name, start, end) inclusive
LABEL_RANGES = [
    (1,  "Phyllostachys_edulis",     1001, 1059),
    (2,  "Aesculus_chinensis",       1060, 1122),
    (3,  "Berberis_anhweiensis",     1552, 1616),
    (4,  "Cercis_chinensis",         1123, 1194),
    (5,  "Indigofera_tinctoria",     1195, 1267),
    (6,  "Acer_palmatum",            1268, 1323),
    (7,  "Phoebe_nammu",             1324, 1385),
    (8,  "Kalopanax_septemlobus",    1386, 1437),
    (9,  "Cinnamomum_japonicum",     1497, 1551),
    (10, "Koelreuteria_paniculata",  1438, 1496),
    (11, "Ilex_macrocarpa",          2001, 2050),
    (12, "Pittosporum_tobira",       2051, 2113),
    (14, "Chimonanthus_praecox",     2114, 2165),
    (15, "Cinnamomum_camphora",      2166, 2230),
    (16, "Viburnum_awabuki",         2231, 2290),
    (17, "Osmanthus_fragrans",       2291, 2346),
    (18, "Cedrus_deodara",           2347, 2423),
    (19, "Ginkgo_biloba",            2424, 2485),
    (20, "Lagerstroemia_indica",     2486, 2546),
    (21, "Nerium_oleander",          2547, 2612),
    (22, "Podocarpus_macrophyllus",  2616, 2675),
    (23, "Prunus_serrulata",         3001, 3055),
    (24, "Ligustrum_lucidum",        3056, 3110),
    (25, "Tonna_sinensis",           3111, 3175),
    (26, "Prunus_persica",           3176, 3229),
    (27, "Manglietia_fordiana",      3230, 3281),
    (28, "Acer_buergerianum",        3282, 3334),
    (29, "Mahonia_bealei",           3335, 3389),
    (30, "Magnolia_grandiflora",     3390, 3446),
    (31, "Populus_canadensis",       3447, 3510),
    (32, "Liriodendron_chinense",    3511, 3563),
    (33, "Citrus_reticulata",        3566, 3621),
   
]

# ======= بناء قاموس للبحث السريع =======
range_to_label = {}
for lbl_id, name, start, end in LABEL_RANGES:
    for n in range(start, end+1):
        range_to_label[str(n).zfill(4)] = name  # keys like "1001"

# ======= Helper Functions: Preprocessing & Augmentation =======

def process_and_save_image(src_path, dest_path, size=IMG_SIZE):
    """Reads image, validates it, resizes it, and saves to dest."""
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB") # Cleaning/Standardizing channels
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(dest_path)
    except Exception as e:
        print(f"Warning: Failed to process {src_path}. Error: {e}")

def get_augmentation_transforms():
    """Returns a simplified augmentation pipeline for offline use."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        # transforms.Resize(IMG_SIZE) # Already resized
    ])

def balance_training_set(train_dir):
    """
    Checks class counts in train_dir.
    Augments minority classes until they match the majority class count.
    """
    print("--- Balancing Training Sets Logic ---")
    classes = [d for d in train_dir.iterdir() if d.is_dir()]
    if not classes:
        return

    # Count files
    class_counts = {c.name: len(list(c.glob("*.jpg"))) for c in classes}
    if not class_counts:
        return
        
    max_count = max(class_counts.values())
    print(f"Max class count: {max_count}. Augmenting others to match...")

    augmentor = get_augmentation_transforms()

    for c in classes:
        current_count = class_counts[c.name]
        diff = max_count - current_count
        if diff <= 0:
            continue
            
        print(f"  Augmenting {c.name}: {current_count} -> {max_count} (+{diff})")
        
        # Get existing images
        existing_files = list(c.glob("*.jpg"))
        
        # Determine how many rounds of full augmentation we need
        # We just sample randomly 'diff' times
        for i in range(diff):
            src_file = random.choice(existing_files)
            try:
                with Image.open(src_file) as img:
                    img = img.convert("RGB")
                    # Augmentation
                    aug_img = augmentor(img)
                    save_name = f"aug_{i}_{src_file.name}"
                    aug_img.save(c / save_name)
            except Exception as e:
                print(f"    Error augmenting {src_file}: {e}")

def create_augmentation_demo(demo_dir=Path("augmentation_demo")):
    """Creates a demonstration grid of augmentations."""
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Pick a random class and random image from RAW_DIR if possible, else skip
    all_files = sorted([p for p in RAW_DIR.glob("*.jpg")])
    if not all_files:
        print("No raw files found for demo.")
        return

    sample_file = random.choice(all_files)
    print(f"Creating augmentation demo using: {sample_file.name}")
    
    augmentor = get_augmentation_transforms()
    
    # Create a grid 1 original + 4 augmented
    try:
        with Image.open(sample_file) as img:
            img = img.convert("RGB").resize(IMG_SIZE)
            
            # Save Original
            img.save(demo_dir / "original.jpg")
            
            # Save 4 augmented versions
            for i in range(1, 5):
                aug_img = augmentor(img)
                aug_img.save(demo_dir / f"aug_ver_{i}.jpg")
                
            print(f"Demonstration images saved to {demo_dir}")
    except Exception as e:
        print(f"Failed to create demo: {e}")


# ======= Main Execution =======

# 1. Collect files
all_files = sorted([p for p in RAW_DIR.glob("*.jpg")])
print(f"Found {len(all_files)} jpg files in {RAW_DIR}")

# Create class folders in OUT_DIR (Temporary holding for split)
classes_names = sorted(set(name for _, name, _, _ in LABEL_RANGES))
for c in classes_names:
    (OUT_DIR / c).mkdir(parents=True, exist_ok=True)

unmatched = []
moved = 0
for p in all_files:
    stem = p.stem  # '1001'
    label = range_to_label.get(stem)
    if label:
        dest = OUT_DIR / label / p.name
        # Preprocessing: Clean & Resize
        process_and_save_image(p, dest)
        moved += 1
    else:
        unmatched.append(p.name)

print(f"Processed & Moved {moved} files. Unmatched: {len(unmatched)}")
if unmatched:
    print("Examples of unmatched files:", unmatched[:20])

# 2. Stratified Split (70/15/15)
def stratified_split_class_folder(class_folder, out_base, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    files = sorted([p for p in class_folder.glob("*.jpg")])
    random.shuffle(files)
    n = len(files)
    n_train = int(floor(n * train_ratio))
    n_val = int(floor(n * val_ratio))
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]
    
    name = class_folder.name
    for subset, lst in [("train", train_files), ("val", val_files), ("test", test_files)]:
        out_dir = out_base / subset / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in lst:
            # Use move to avoid keeping temp files
            shutil.move(str(f), str(out_dir / f.name)) 

# Apply split
print("Splitting dataset...")
for c in classes_names:
    class_folder = OUT_DIR / c
    if class_folder.exists():
        stratified_split_class_folder(class_folder, OUT_DIR, 0.7, 0.15, 0.15)
        # Remove empty temp folder
        try:
            class_folder.rmdir()
        except:
            pass

print("Finished creating train/val/test splits under:", OUT_DIR)

# 3. Handle Imbalanced Data (Augmentation) -> ONLY on TRAIN set
balance_training_set(OUT_DIR / "train")

# 4. Demonstation
create_augmentation_demo(OUT_DIR / "augmentation_demo")
