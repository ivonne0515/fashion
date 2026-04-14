
import os
import json
import time
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.amp import GradScaler, autocast

import torchvision.transforms as T
from torchvision import models

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit


# CONFIGURACION

DATA_ROOT       = Path(r"C:\Users\USUARIO\Documents\fashion_deep")
IMAGES_DIR      = DATA_ROOT / "train"
SPLITS_DIR      = DATA_ROOT / "splits"
CHECKPOINT_DIR  = Path("./checkpoints")
RESULTS_DIR     = Path("./results")

SUBSET_FRACTION = 0.15
IMG_SIZE        = 224
BATCH_SIZE      = 48
NUM_WORKERS     = 4
EPOCHS          = 15
SEED            = 42
COMPILE_MODEL   = False

# Early stopping mas permisivo
PATIENCE        = 5
MIN_EPOCH       = 6
MIN_DELTA       = 1e-4       r

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

NUM_CLASSES = None
CLASS_NAMES = None



# 1. REPRODUCIBILIDAD Y DEVICE

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 2. DATASET CUSTOM CON CACHE EN RAM

class iMaterialistDataset(Dataset):


    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.targets = self.df['class_idx'].tolist()
        self._cached_images = None

    def preload(self, label=""):
        print(f"  Precargando {len(self.df):,} imagenes a RAM {label}...")
        t0 = time.time()
        self._cached_images = []
        for img_id in self.df['image_id']:
            img = Image.open(self.images_dir / str(img_id)).convert('RGB')
            # Pre-redimensionar para acelerar el transform posterior
            img.thumbnail((288, 288), Image.BILINEAR)
            # .copy() cierra el file handle y copia los pixeles
            self._cached_images.append(img.copy())
        mb = sum(i.size[0] * i.size[1] * 3 for i in self._cached_images) / 1e6
        print(f"    Listo en {time.time()-t0:.1f}s (~{mb:.0f} MB en RAM)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self._cached_images is not None:
            img = self._cached_images[idx]
        else:
            img_id = self.df.iloc[idx]['image_id']
            img = Image.open(self.images_dir / str(img_id)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(self.df.iloc[idx]['class_idx'])



# 3. VERIFICACION + CARGA DE METADATA

def verify_dataset():
    global NUM_CLASSES, CLASS_NAMES

    print("\n" + "=" * 60)
    print("  VERIFICACION DEL DATASET")
    print("=" * 60)

    required_files = [
        SPLITS_DIR / 'train.csv',
        SPLITS_DIR / 'val.csv',
        SPLITS_DIR / 'test.csv',
        SPLITS_DIR / 'class_names.json',
    ]
    missing = [p for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"No se encontraron archivos necesarios:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nPrimero ejecuta: python prepare_dataset.py"
        )

    with open(SPLITS_DIR / 'class_names.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    NUM_CLASSES = meta['num_classes']
    CLASS_NAMES = meta['class_names']

    print(f"  NUM_CLASSES: {NUM_CLASSES}")
    print(f"  CLASS_NAMES: {CLASS_NAMES}")

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de imagenes: {IMAGES_DIR}")

    for split in ['train', 'val', 'test']:
        df = pd.read_csv(SPLITS_DIR / f'{split}.csv')
        counts = Counter(df['class_idx'].tolist())
        n_subset = int(len(df) * SUBSET_FRACTION)
        print(f"\n  [{split.upper()}]  {len(df):,} imagenes  -> 15% = {n_subset:,}")
        min_per_class = min(counts.values())
        print(f"     clase minima: {min_per_class:,} imagenes "
              f"({int(min_per_class * SUBSET_FRACTION)} en subset)")

    print("\n  Verificacion OK.\n")



# 4. TRANSFORMS - RandomErasing va al final (requiere tensor)

def get_transforms(split='train', augment_mode='standard'):
    if split == 'train':
        pil_augs = [
            T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=15),
        ]
        if augment_mode == 'specialized':
            pil_augs.append(T.ColorJitter(brightness=0.2, contrast=0.2,
                                          saturation=0.3, hue=0.1))
            tensor_augs = [T.RandomErasing(p=0.5, scale=(0.02, 0.33))]
        else:
            pil_augs.append(T.ColorJitter(brightness=0.2, contrast=0.2))
            tensor_augs = []

        return T.Compose(
            pil_augs
            + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
            + tensor_augs
        )
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])



# 5. SUBSET + PRECARGA

def stratified_subset(dataset, fraction, seed=SEED):
    targets = np.array(dataset.targets)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=fraction,
                                      random_state=seed)
    idx, _ = next(splitter.split(np.zeros(len(targets)), targets))
    return Subset(dataset, idx.tolist())


def _apply_subset_and_preload(base_dataset, subset, label):
    """Reduce el dataset base a solo los indices del subset y precarga
    esas imagenes a RAM. Devuelve un Subset nuevo con indices 0..N-1."""
    base_dataset.df = base_dataset.df.iloc[subset.indices].reset_index(drop=True)
    base_dataset.targets = base_dataset.df['class_idx'].tolist()
    base_dataset.preload(label=label)
    return Subset(base_dataset, list(range(len(base_dataset.df))))


def load_datasets(augment_mode='standard'):
    print(f"Cargando splits desde {SPLITS_DIR}")

    train_full = iMaterialistDataset(SPLITS_DIR / 'train.csv', IMAGES_DIR,
                                     transform=get_transforms('train', augment_mode))
    val_full   = iMaterialistDataset(SPLITS_DIR / 'val.csv', IMAGES_DIR,
                                     transform=get_transforms('val'))
    test_full  = iMaterialistDataset(SPLITS_DIR / 'test.csv', IMAGES_DIR,
                                     transform=get_transforms('test'))

    train_sub = stratified_subset(train_full, SUBSET_FRACTION)
    val_sub   = stratified_subset(val_full,   SUBSET_FRACTION)
    test_sub  = stratified_subset(test_full,  SUBSET_FRACTION)

    train_ds = _apply_subset_and_preload(train_full, train_sub, f"(train-{augment_mode})")
    val_ds   = _apply_subset_and_preload(val_full,   val_sub,   "(val)")
    test_ds  = _apply_subset_and_preload(test_full,  test_sub,  "(test)")

    return train_ds, val_ds, test_ds


def make_loader(ds, shuffle=True, batch_size=BATCH_SIZE):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=shuffle,
    )



# 6. MODELOS

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class M0_CNN_Scratch(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_M1_frozen(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512), nn.ReLU(inplace=True),
        nn.Linear(512, num_classes)
    )
    return model


def build_M2_finetuned(num_classes, freeze_phase=False):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if freeze_phase:
        for p in model.parameters():
            p.requires_grad = False
    else:
        for name, p in model.named_parameters():
            if any(k in name for k in ['layer1', 'layer2', 'conv1', 'bn1']):
                p.requires_grad = False
            else:
                p.requires_grad = True
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 512), nn.ReLU(inplace=True),
        nn.Linear(512, num_classes)
    )
    return model


class SEBlock(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        mid = max(channels // ratio, 4)
        self.squeeze    = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid()
        )

    def forward(self, x):
        s = self.squeeze(x)
        e = self.excitation(s).view(x.size(0), -1, 1, 1)
        return x * e


class SEBottleneck(nn.Module):
    expansion = 4
    def __init__(self, orig, se_ratio=16):
        super().__init__()
        channels = orig.conv3.in_channels
        self.conv1, self.bn1 = orig.conv1, orig.bn1
        self.conv2, self.bn2 = orig.conv2, orig.bn2
        self.conv3, self.bn3 = orig.conv3, orig.bn3
        self.relu = orig.relu
        self.se   = SEBlock(channels, ratio=se_ratio)
        self.downsample = orig.downsample
        self.stride     = orig.stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def insert_se_blocks(resnet, se_ratio=16):
    for ln in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(resnet, ln)
        for i, b in enumerate(layer):
            layer[i] = SEBottleneck(b, se_ratio=se_ratio)
    return resnet


class MultiTaskHead(nn.Module):
    def __init__(self, in_feats, n_type, n_color=12, n_material=8):
        super().__init__()
        self.head_type     = nn.Sequential(nn.Dropout(0.3),
                                           nn.Linear(in_feats, 256),
                                           nn.ReLU(), nn.Linear(256, n_type))
        self.head_color    = nn.Sequential(nn.Dropout(0.3),
                                           nn.Linear(in_feats, 128),
                                           nn.ReLU(), nn.Linear(128, n_color))
        self.head_material = nn.Sequential(nn.Dropout(0.3),
                                           nn.Linear(in_feats, 128),
                                           nn.ReLU(), nn.Linear(128, n_material))
        self.head_pattern  = nn.Sequential(nn.Dropout(0.3),
                                           nn.Linear(in_feats, 64),
                                           nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return (self.head_type(x), self.head_color(x),
                self.head_material(x), self.head_pattern(x))


class M3_SEResNet50_MultiTask(nn.Module):
    def __init__(self, num_classes, se_ratio=16):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone = insert_se_blocks(backbone, se_ratio=se_ratio)
        in_f = backbone.fc.in_features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.flatten  = nn.Flatten()
        self.heads    = MultiTaskHead(in_feats=in_f, n_type=num_classes)

    def forward(self, x):
        return self.heads(self.flatten(self.backbone(x)))


def build_M4(num_classes):
    return M3_SEResNet50_MultiTask(num_classes)



# 7. TRAINING UTILITIES

class LabelSmoothingCE(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_prob,
                                     self.smoothing / (self.num_classes - 1))
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return (-smooth * log_prob).sum(dim=-1).mean()


def cutmix_batch(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    B   = images.size(0)
    idx = torch.randperm(B, device=images.device)
    H, W = images.size(2), images.size(3)
    cut_ratio = np.sqrt(1 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam_adj = 1 - (x2 - x1) * (y2 - y1) / (H * W)
    return mixed, labels, labels[idx], lam_adj


def multitask_loss(outputs, labels, ce_fn,
                   lambdas=(0.50, 0.20, 0.20, 0.10)):
    type_out, color_out, mat_out, pat_out = outputs
    dummy_color = torch.zeros(labels.size(0), dtype=torch.long, device=labels.device)
    dummy_mat   = torch.zeros(labels.size(0), dtype=torch.long, device=labels.device)
    dummy_pat   = torch.zeros(labels.size(0), 1, device=labels.device)
    l0 = ce_fn(type_out, labels)
    l1 = nn.CrossEntropyLoss()(color_out, dummy_color)
    l2 = nn.CrossEntropyLoss()(mat_out,   dummy_mat)
    l3 = nn.BCEWithLogitsLoss()(pat_out, dummy_pat)
    return lambdas[0]*l0 + lambdas[1]*l1 + lambdas[2]*l2 + lambdas[3]*l3


def train_epoch(model, loader, optimizer, criterion, scaler,
                multitask=False, use_cutmix=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(DEVICE, non_blocking=True)

        if use_cutmix and random.random() > 0.5:
            images, lab_a, lab_b, lam = cutmix_batch(images, labels)
        else:
            lab_a = lab_b = labels
            lam = 1.0

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            if multitask:
                outputs = model(images)
                loss = (lam       * multitask_loss(outputs, lab_a, criterion)
                        + (1-lam) * multitask_loss(outputs, lab_b, criterion))
                preds = outputs[0].argmax(1)
            else:
                outputs = model(images)
                loss = (lam       * criterion(outputs, lab_a)
                        + (1-lam) * criterion(outputs, lab_b))
                preds = outputs.argmax(1)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, multitask=False):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast('cuda'):
            if multitask:
                outputs = model(images)
                loss = multitask_loss(outputs, labels, criterion)
                preds = outputs[0].argmax(1)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(1)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return total_loss / n, acc, macro_f1, np.array(all_preds), np.array(all_labels)


def train_model(model, train_loader, val_loader, model_name,
                epochs=EPOCHS, lr=1e-3, multitask=False, use_cutmix=False,
                scheduler_type='cosine', patience=PATIENCE,
                min_epoch=MIN_EPOCH, min_delta=MIN_DELTA,
                compile_model=COMPILE_MODEL):
    print(f"\n{'='*60}\n  {model_name}\n{'='*60}")

    model = model.to(DEVICE, memory_format=torch.channels_last)

    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            print(f"  AVISO: torch.compile fallo ({e}); continuando sin compilar")

    criterion = LabelSmoothingCE(NUM_CLASSES, smoothing=0.1)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4)

    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    scaler = GradScaler('cuda')
    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_f1, wait, best_state = 0.0, 0, None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer,
                                      criterion, scaler, multitask, use_cutmix)
        vl_loss, vl_acc, vl_f1, _, _ = eval_epoch(model, val_loader,
                                                  criterion, multitask)
        scheduler.step()

        for k, v in zip(['train_loss','val_loss','train_acc','val_acc','val_f1'],
                        [tr_loss, vl_loss, tr_acc, vl_acc, vl_f1]):
            history[k].append(v)

        print(f"Ep {epoch:03d}/{epochs} | "
              f"Tr L {tr_loss:.4f} A {tr_acc:.3f} | "
              f"Val L {vl_loss:.4f} A {vl_acc:.3f} F1 {vl_f1:.3f} | "
              f"{time.time()-t0:.1f}s")

        if vl_f1 > best_f1 + min_delta:
            best_f1 = vl_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience and epoch >= min_epoch:
                print(f"  Early stopping @ epoch {epoch}  (best F1={best_f1:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# 8. MAIN

def main():
    print(f"Device:      {DEVICE}")
    print(f"PyTorch:     {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        print(f"VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    verify_dataset()

    train_ds, val_ds, test_ds = load_datasets()
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)
    test_loader  = make_loader(test_ds,  shuffle=False)

    print(f"\nTrain: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"Batches/epoch (train): {len(train_loader)}")

    train_ds_spec, _, _ = load_datasets(augment_mode='specialized')
    train_loader_spec = make_loader(train_ds_spec, shuffle=True)

    # M0
    set_seed()
    m0 = M0_CNN_Scratch(NUM_CLASSES)
    m0, hist_m0 = train_model(m0, train_loader, val_loader,
                              'M0 - CNN from Scratch', lr=1e-3)

    # M1
    set_seed()
    m1 = build_M1_frozen(NUM_CLASSES)
    m1, hist_m1 = train_model(m1, train_loader, val_loader,
                              'M1 - ResNet-50 Frozen', lr=1e-3)

    # M2
    set_seed()
    m2 = build_M2_finetuned(NUM_CLASSES, freeze_phase=False)
    m2, hist_m2 = train_model(m2, train_loader, val_loader,
                              'M2 - ResNet-50 Fine-tuned', lr=1e-4)

    # M3
    set_seed()
    m3 = M3_SEResNet50_MultiTask(NUM_CLASSES)
    m3, hist_m3 = train_model(m3, train_loader, val_loader,
                              'M3 - SE-ResNet-50 + Multitask',
                              lr=1e-4, multitask=True)

    # M4
    set_seed()
    m4 = build_M4(NUM_CLASSES)
    m4, hist_m4 = train_model(m4, train_loader_spec, val_loader,
                              'M4 - SE-ResNet-50 + Aug Especializado',
                              lr=1e-4, multitask=True, use_cutmix=True)

    # Evaluacion final
    criterion_eval = LabelSmoothingCE(NUM_CLASSES, 0.1)
    results = {}
    for name, model, multitask in [
        ('M0 CNN Scratch',     m0, False),
        ('M1 ResNet Frozen',   m1, False),
        ('M2 ResNet Finetune', m2, False),
        ('M3 SE-ResNet MT',    m3, True),
        ('M4 SE-ResNet Aug',   m4, True),
    ]:
        _, acc, f1, preds, labels_arr = eval_epoch(model, test_loader,
                                                   criterion_eval, multitask)
        results[name] = {'acc': acc, 'f1': f1,
                         'preds': preds, 'labels': labels_arr}
        print(f"{name:<22} | Test Acc: {acc:.4f} | Macro F1: {f1:.4f}")

    torch.save(m4.state_dict(), CHECKPOINT_DIR / 'M4_SE_ResNet50_best.pth')
    torch.save(m2.state_dict(), CHECKPOINT_DIR / 'M2_ResNet50_baseline.pth')
    summary = {k: {'acc': float(v['acc']), 'f1': float(v['f1'])}
               for k, v in results.items()}
    with open(CHECKPOINT_DIR / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nModelos y resumen guardados en checkpoints/")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()