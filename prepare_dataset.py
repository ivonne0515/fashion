
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


# CONFIG

DATA_ROOT     = Path(r"C:\Users\USUARIO\Documents\fashion_deep")
TRAIN_CSV     = DATA_ROOT / "train.csv"
MIN_SAMPLES_PER_CLASS = 100     # descarta clases con menos ejemplos
LABELS_JSON   = DATA_ROOT / "label_descriptions.json"
OUT_DIR       = DATA_ROOT / "splits"
OUT_DIR.mkdir(exist_ok=True)


KEEP_CATEGORIES_MAX = 26

TEST_FRACTION = 0.15
VAL_FRACTION  = 0.15
SEED          = 42



# 1. Cargar nombres de clases

print("Cargando label_descriptions.json...")
with open(LABELS_JSON, 'r', encoding='utf-8') as f:
    label_desc = json.load(f)

all_categories = {c['id']: c['name'] for c in label_desc['categories']}
print(f"  Total categorias en el dataset: {len(all_categories)}")

kept_ids = [cid for cid in sorted(all_categories.keys())
            if cid <= KEEP_CATEGORIES_MAX]
print(f"  Categorias a usar (0-{KEEP_CATEGORIES_MAX}): {len(kept_ids)}")



# 2. Leer train.csv (parte lenta, ~30-60s)

print(f"\nLeyendo {TRAIN_CSV} ({TRAIN_CSV.stat().st_size / 1e9:.2f} GB)...")
t0 = time.time()

# Solo necesitamos 3 columnas; usecols ahorra memoria
df = pd.read_csv(TRAIN_CSV, usecols=['ImageId', 'EncodedPixels', 'ClassId'],
                 dtype={'ImageId': str, 'EncodedPixels': str, 'ClassId': str})
print(f"  {len(df):,} filas cargadas en {time.time()-t0:.1f}s")



# 3. Extraer categoria principal y calcular area de cada segmento

print("\nProcesando ClassId y calculando areas...")
t0 = time.time()


df['category'] = df['ClassId'].str.split('_').str[0].astype(int)


before = len(df)
df = df[df['category'] <= KEEP_CATEGORIES_MAX].copy()
print(f"  Filas filtradas por categoria: {before:,} -> {len(df):,}")


def rle_area(rle_str):
    if not isinstance(rle_str, str):
        return 0
    parts = rle_str.split()
    return sum(int(parts[i]) for i in range(1, len(parts), 2))

df['area'] = df['EncodedPixels'].map(rle_area)
print(f"  Areas calculadas en {time.time()-t0:.1f}s")


# 4. Para cada imagen, quedarse con la prenda de mayor area

print("\nSeleccionando prenda dominante por imagen...")
t0 = time.time()


idx_max = df.groupby('ImageId')['area'].idxmax()
dominant = df.loc[idx_max, ['ImageId', 'category']].reset_index(drop=True)
print(f"  {len(dominant):,} imagenes unicas en {time.time()-t0:.1f}s")


# 5. Filtrar clases con muy pocas imagenes y remapear a indices contiguos

cat_counts = dominant['category'].value_counts()
kept_cats = sorted(cat_counts[cat_counts >= MIN_SAMPLES_PER_CLASS].index.tolist())
dropped = sorted(cat_counts[cat_counts < MIN_SAMPLES_PER_CLASS].index.tolist())

if dropped:
    print(f"\nDescartando clases con < {MIN_SAMPLES_PER_CLASS} imagenes:")
    for cid in dropped:
        print(f"  id {cid:2d}  {all_categories[cid]:<40} {cat_counts[cid]:>5}")

before = len(dominant)
dominant = dominant[dominant['category'].isin(kept_cats)].copy()
print(f"\nImagenes tras filtrado: {before:,} -> {len(dominant):,}")


cat_to_idx  = {c: i for i, c in enumerate(kept_cats)}
idx_to_name = [all_categories[c] for c in kept_cats]
dominant['class_idx'] = dominant['category'].map(cat_to_idx)

print(f"\nClases finales ({len(kept_cats)}):")
counts = dominant['class_idx'].value_counts().sort_index()
for i, name in enumerate(idx_to_name):
    print(f"  {i:2d}  {name:<40} {counts[i]:>6,}")

present_cats = kept_cats  # para compatibilidad con el JSON de metadata

# 6. Split train/val/test estratificado 70/15/15
print("\nGenerando splits estratificados 70/15/15...")
y = dominant['class_idx'].values


sss1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                              random_state=SEED)
trainval_idx, test_idx = next(sss1.split(np.zeros(len(y)), y))


y_trainval = y[trainval_idx]
val_rel_frac = VAL_FRACTION / (1 - TEST_FRACTION)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel_frac,
                              random_state=SEED)
train_rel, val_rel = next(sss2.split(np.zeros(len(y_trainval)), y_trainval))

train_idx = trainval_idx[train_rel]
val_idx   = trainval_idx[val_rel]

print(f"  Train: {len(train_idx):,}")
print(f"  Val:   {len(val_idx):,}")
print(f"  Test:  {len(test_idx):,}")



# 7. Guardar CSVs y metadata

def save_split(indices, name):
    sub = dominant.iloc[indices][['ImageId', 'class_idx']].copy()
    sub.columns = ['image_id', 'class_idx']
    path = OUT_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"  {path}  ({len(sub):,} filas)")

print("\nGuardando splits...")
save_split(train_idx, 'train')
save_split(val_idx,   'val')
save_split(test_idx,  'test')

with open(OUT_DIR / 'class_names.json', 'w', encoding='utf-8') as f:
    json.dump({
        'num_classes': len(present_cats),
        'class_names': idx_to_name,
        'original_category_ids': present_cats,
    }, f, indent=2, ensure_ascii=False)
print(f"  {OUT_DIR / 'class_names.json'}")

print("\nListo. Ahora ejecuta fashion.py.")