# scripts/prepare_flavia.py
import os
import shutil
import random
from pathlib import Path
from math import floor

random.seed(42)

# ======= ضبط المسارات =======
RAW_DIR = Path("data/Leaves")   # بعد فك الضغط
OUT_DIR = Path("data/flavia")      # ستكون البنية هنا
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    # NOTE: check if there are other labels in your downloaded dataset (some labels may be missing)
]

# ======= بناء قاموس للبحث السريع =======
range_to_label = {}
for lbl_id, name, start, end in LABEL_RANGES:
    for n in range(start, end+1):
        range_to_label[str(n).zfill(4)] = name  # keys like "1001"

# ======= اجمع الملفات ونقلها =======
all_files = sorted([p for p in RAW_DIR.glob("*.jpg")])
print(f"Found {len(all_files)} jpg files in {RAW_DIR}")

# Create class folders
classes = sorted(set(name for _, name, _, _ in LABEL_RANGES))
for c in classes:
    (OUT_DIR / c).mkdir(parents=True, exist_ok=True)

unmatched = []
moved = 0
for p in all_files:
    stem = p.stem  # '1001'
    label = range_to_label.get(stem)
    if label:
        dest = OUT_DIR / label / p.name
        shutil.copy2(p, dest)   # copy, keep raw intact
        moved += 1
    else:
        unmatched.append(p.name)

print(f"Moved {moved} files. Unmatched: {len(unmatched)}")
if unmatched:
    print("Examples of unmatched files:", unmatched[:20])

# ======= الآن نعمل split stratified (70/15/15) على كل class folder =======
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
            shutil.copy2(f, out_dir / f.name)

# apply to each class
for c in classes:
    class_folder = OUT_DIR / c
    stratified_split_class_folder(class_folder, OUT_DIR, 0.7, 0.15, 0.15)

print("Finished creating train/val/test splits under:", OUT_DIR)
