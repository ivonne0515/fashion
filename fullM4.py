import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler

# Importamos todo del script principal
from fashion import (
    DEVICE, SPLITS_DIR, IMAGES_DIR, CHECKPOINT_DIR,
    BATCH_SIZE, NUM_WORKERS, SEED,
    PATIENCE, MIN_EPOCH, MIN_DELTA,
    iMaterialistDataset, get_transforms, stratified_subset,
    _apply_subset_and_preload,
    build_M4, LabelSmoothingCE, train_epoch, eval_epoch,
    set_seed,
)

# CONFIGURACION ESPECIFICA PARA ESTE RUN
EPOCHS      = 15
LR          = 1e-4
FULL_FRACTION = 1.0    # usar el 100% del dataset
# BATCH_SIZE y NUM_WORKERS se heredan de fashion.py


def load_full_datasets():
    """Carga el dataset completo (100%) con transforms specialized para M4."""
    print(f"Cargando splits completos desde {SPLITS_DIR}")
    print("(precarga en RAM, puede tardar ~13 minutos)")

    train_full = iMaterialistDataset(SPLITS_DIR / 'train.csv', IMAGES_DIR,
                                     transform=get_transforms('train', 'specialized'))
    val_full   = iMaterialistDataset(SPLITS_DIR / 'val.csv', IMAGES_DIR,
                                     transform=get_transforms('val'))
    test_full  = iMaterialistDataset(SPLITS_DIR / 'test.csv', IMAGES_DIR,
                                     transform=get_transforms('test'))

    # Subset al 100% (stratified_subset con fraction=1.0 devuelve todo)
    print(f"  Precargando {len(train_full):,} imagenes a RAM (train-FULL)...")
    t0 = time.time()
    train_full.preload(label="(train-FULL)")
    val_full.preload(label="(val-FULL)")
    test_full.preload(label="(test-FULL)")

    train_ds = train_full
    val_ds   = val_full
    test_ds  = test_full

    return train_ds, val_ds, test_ds


def make_loader(ds, shuffle=True):
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=shuffle,
    )


def main():
    print(f"Device:      {DEVICE}")
    print(f"PyTorch:     {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
        print(f"VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    with open(SPLITS_DIR / 'class_names.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    NUM_CLASSES = meta['num_classes']
    CLASS_NAMES = meta['class_names']
    print(f"\nNUM_CLASSES: {NUM_CLASSES}")
    print(f"CLASSES: {CLASS_NAMES}")

    t_precarga = time.time()
    train_ds, val_ds, test_ds = load_full_datasets()
    print(f"\nPrecarga completa en {(time.time()-t_precarga)/60:.1f} min")

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)
    test_loader  = make_loader(test_ds,  shuffle=False)

    print(f"\nTrain: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"Batches/epoch (train): {len(train_loader)}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}")

    print("\n" + "=" * 60)
    print("  M4 - SE-ResNet-50 + Aug Especializado (DATASET COMPLETO)")
    print("=" * 60)

    set_seed()
    model = build_M4(NUM_CLASSES)
    model = model.to(DEVICE, memory_format=torch.channels_last)

    criterion = LabelSmoothingCE(NUM_CLASSES, smoothing=0.1)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler('cuda')

    history = {'train_loss': [], 'val_loss': [],
               'train_acc': [], 'val_acc': [], 'val_f1': []}
    best_f1, wait, best_state = 0.0, 0, None
    t_total = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer,
                                      criterion, scaler,
                                      multitask=True, use_cutmix=True)
        vl_loss, vl_acc, vl_f1, _, _ = eval_epoch(model, val_loader,
                                                  criterion, multitask=True)
        scheduler.step()

        for k, v in zip(['train_loss','val_loss','train_acc','val_acc','val_f1'],
                        [tr_loss, vl_loss, tr_acc, vl_acc, vl_f1]):
            history[k].append(v)

        elapsed = time.time() - t0
        print(f"Ep {epoch:03d}/{EPOCHS} | "
              f"Tr L {tr_loss:.4f} A {tr_acc:.3f} | "
              f"Val L {vl_loss:.4f} A {vl_acc:.3f} F1 {vl_f1:.3f} | "
              f"{elapsed:.1f}s ({elapsed/60:.1f} min)")

        if vl_f1 > best_f1 + MIN_DELTA:
            best_f1 = vl_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            # Guardado intermedio por si el proceso se interrumpe
            torch.save(best_state, CHECKPOINT_DIR / 'M4_full.pth')
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE and epoch >= MIN_EPOCH:
                print(f"  Early stopping @ epoch {epoch}  (best F1={best_f1:.4f})")
                break

    total_min = (time.time() - t_total) / 60
    print(f"\nEntrenamiento total: {total_min:.1f} min")

    if best_state:
        model.load_state_dict(best_state)


    print("\nEvaluando en test...")
    _, test_acc, test_f1, preds, labels_arr = eval_epoch(
        model, test_loader, criterion, multitask=True)
    print(f"  Test Acc: {test_acc:.4f}")
    print(f"  Test F1:  {test_f1:.4f}")

    with open(CHECKPOINT_DIR / 'M4_full_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    with open(CHECKPOINT_DIR / 'M4_full_results.json', 'w') as f:
        json.dump({
            'test_acc': float(test_acc),
            'test_f1':  float(test_f1),
            'best_val_f1': float(best_f1),
            'epochs_trained': len(history['train_loss']),
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'total_minutes': round(total_min, 1),
            'train_size': len(train_ds),
            'val_size':   len(val_ds),
            'test_size':  len(test_ds),
        }, f, indent=2)

    print(f"\nArchivos guardados:")
    print(f"  {CHECKPOINT_DIR / 'M4_full.pth'}")
    print(f"  {CHECKPOINT_DIR / 'M4_full_results.json'}")
    print(f"  {CHECKPOINT_DIR / 'M4_full_history.json'}")

    try:
        with open(CHECKPOINT_DIR / 'results_summary.json', 'r') as f:
            old_summary = json.load(f)
        old_m4 = old_summary.get('M4 SE-ResNet Aug', {})
        if old_m4:
            print("\n" + "=" * 60)
            print("  COMPARACION M4 15% vs M4 100%")
            print("=" * 60)
            print(f"  M4 (15%):   Acc={old_m4['acc']:.4f}  F1={old_m4['f1']:.4f}")
            print(f"  M4 (100%):  Acc={test_acc:.4f}  F1={test_f1:.4f}")
            print(f"  Delta Acc:  {test_acc - old_m4['acc']:+.4f}")
            print(f"  Delta F1:   {test_f1 - old_m4['f1']:+.4f}")
    except FileNotFoundError:
        pass


if __name__ == '__main__':
    main()