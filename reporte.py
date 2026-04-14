
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (confusion_matrix, classification_report,
                              f1_score, accuracy_score)


from fashion import (
    DEVICE, SPLITS_DIR, IMAGES_DIR, CHECKPOINT_DIR, BATCH_SIZE, NUM_WORKERS,
    iMaterialistDataset, get_transforms, stratified_subset,
    _apply_subset_and_preload,
    build_M2_finetuned, M3_SEResNet50_MultiTask, build_M4,
    LabelSmoothingCE, eval_epoch, SUBSET_FRACTION,
)

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")


def plot_confusion(preds, labels, class_names, title, path):
    cm = confusion_matrix(labels, preds, normalize='true',
                          labels=list(range(len(class_names))))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proporcion'}, annot_kws={'size': 8})
    plt.xlabel('Predicho', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {path}")


def evaluate_saved_model(model, ckpt_path, multitask, label,
                         test_loader, criterion):
    """Carga pesos y evalua en test."""
    print(f"\n  Evaluando {label}...")
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE, memory_format=torch.channels_last)
    _, acc, f1, preds, labels = eval_epoch(model, test_loader,
                                           criterion, multitask)
    print(f"    Acc={acc:.4f}  F1={f1:.4f}")
    return preds, labels


def main():

    # 1. Cargar resumen de resultados

    with open(CHECKPOINT_DIR / 'results_summary.json', 'r') as f:
        summary = json.load(f)

    with open(SPLITS_DIR / 'class_names.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    CLASS_NAMES = meta['class_names']
    NUM_CLASSES = meta['num_classes']

    print("Resumen cargado:")
    for name, m in summary.items():
        print(f"  {name:<22} Acc={m['acc']:.4f}  F1={m['f1']:.4f}")


    # 2. Tabla resumen (CSV)

    df_summary = pd.DataFrame([
        {
            'Model': name,
            'Test Accuracy': round(m['acc'], 4),
            'Macro F1': round(m['f1'], 4),
        }
        for name, m in summary.items()
    ])
    csv_path = RESULTS_DIR / 'summary_table.csv'
    df_summary.to_csv(csv_path, index=False)
    print(f"\nTabla guardada: {csv_path}")
    print(df_summary.to_string(index=False))


    # 3. Grafico de barras: Acc y F1 por modelo

    fig, ax = plt.subplots(figsize=(11, 5.5))
    names = list(summary.keys())
    accs = [summary[n]['acc'] for n in names]
    f1s  = [summary[n]['f1']  for n in names]

    x = np.arange(len(names))
    width = 0.36
    bars1 = ax.bar(x - width/2, accs, width, label='Test Accuracy',
                   color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + width/2, f1s, width, label='Macro F1',
                   color='#2ecc71', edgecolor='white')

    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                    f'{b.get_height():.3f}', ha='center', va='bottom',
                    fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n', 1) for n in names], fontsize=10)
    ax.set_ylabel('Score')
    ax.set_ylim(0, max(accs + f1s) * 1.15)
    ax.set_title('Comparacion de modelos - Test Accuracy & Macro F1',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'barchart_models.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grafico guardado: {RESULTS_DIR / 'barchart_models.png'}")


    # 4. Re-evaluar M2 y M4

    print("\nCargando test set para re-evaluar M2 y M4...")

    test_full = iMaterialistDataset(SPLITS_DIR / 'test.csv', IMAGES_DIR,
                                    transform=get_transforms('test'))
    test_sub = stratified_subset(test_full, SUBSET_FRACTION)
    test_ds = _apply_subset_and_preload(test_full, test_sub, "(test)")

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             persistent_workers=True, prefetch_factor=4)

    criterion = LabelSmoothingCE(NUM_CLASSES, 0.1)

    # M2
    m2 = build_M2_finetuned(NUM_CLASSES, freeze_phase=False)
    preds_m2, labels_m2 = evaluate_saved_model(
        m2, CHECKPOINT_DIR / 'M2_ResNet50_baseline.pth', False, 'M2',
        test_loader, criterion)

    # M4
    m4 = build_M4(NUM_CLASSES)
    preds_m4, labels_m4 = evaluate_saved_model(
        m4, CHECKPOINT_DIR / 'M4_SE_ResNet50_best.pth', True, 'M4',
        test_loader, criterion)


    # 5. Matrices de confusion

    print("\nGenerando matrices de confusion...")
    plot_confusion(preds_m2, labels_m2, CLASS_NAMES,
                   'Matriz de Confusion - M2 ResNet Finetune (normalizada)',
                   RESULTS_DIR / 'confusion_m2.png')
    plot_confusion(preds_m4, labels_m4, CLASS_NAMES,
                   'Matriz de Confusion - M4 SE-ResNet + Aug (normalizada)',
                   RESULTS_DIR / 'confusion_m4.png')


    # 6. F1 por clase: M2 vs M4

    print("\nCalculando F1 por clase...")
    f1_per_class_m2 = f1_score(labels_m2, preds_m2, average=None,
                                labels=list(range(len(CLASS_NAMES))),
                                zero_division=0)
    f1_per_class_m4 = f1_score(labels_m4, preds_m4, average=None,
                                labels=list(range(len(CLASS_NAMES))),
                                zero_division=0)

    order = np.argsort(-f1_per_class_m4)
    names_ord = [CLASS_NAMES[i] for i in order]
    m2_ord = f1_per_class_m2[order]
    m4_ord = f1_per_class_m4[order]

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(names_ord))
    h = 0.4
    ax.barh(y - h/2, m2_ord, h, label='M2', color='#3498db', edgecolor='white')
    ax.barh(y + h/2, m4_ord, h, label='M4', color='#e67e22', edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels(names_ord, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('F1-Score')
    ax.set_xlim(0, 1.0)
    ax.set_title('F1 por clase - M2 vs M4', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    for i, (v_m2, v_m4) in enumerate(zip(m2_ord, m4_ord)):
        ax.text(v_m2 + 0.01, i - h/2, f'{v_m2:.2f}', va='center', fontsize=8)
        ax.text(v_m4 + 0.01, i + h/2, f'{v_m4:.2f}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'per_class_f1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {RESULTS_DIR / 'per_class_f1.png'}")


    # 7. Reportes de clasificacion

    for preds, labels, name in [(preds_m2, labels_m2, 'm2'),
                                (preds_m4, labels_m4, 'm4')]:
        report = classification_report(labels, preds, target_names=CLASS_NAMES,
                                       digits=4, zero_division=0)
        path = RESULTS_DIR / f'classification_report_{name}.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  Guardado: {path}")


    # 8. Test de McNemar (M2 vs M4)

    print("\nTest de McNemar (M2 vs M4):")
    try:
        from scipy.stats import binomtest
        labels_arr = labels_m2  # son los mismos para ambos
        m2_correct = preds_m2 == labels_arr
        m4_correct = preds_m4 == labels_arr
        n00 = int(((~m2_correct) & (~m4_correct)).sum())
        n01 = int(((~m2_correct) &   m4_correct).sum())
        n10 = int((  m2_correct  & (~m4_correct)).sum())
        n11 = int((  m2_correct  &   m4_correct).sum())
        n = n01 + n10
        if n > 0:
            k = min(n01, n10)
            p = 2 * binomtest(k, n, 0.5).pvalue
            p = min(p, 1.0)
        else:
            p = 1.0
        print(f"  M2 correcto, M4 correcto:   {n11}")
        print(f"  M2 correcto, M4 incorrecto: {n10}")
        print(f"  M2 incorrecto, M4 correcto: {n01}")
        print(f"  Ambos incorrectos:          {n00}")
        print(f"  McNemar p-valor: {p:.4f}  "
              f"{'(significativo, p<0.05)' if p < 0.05 else '(no significativo)'}")
        with open(RESULTS_DIR / 'mcnemar_m2_m4.txt', 'w') as f:
            f.write(f"McNemar test M2 vs M4\n")
            f.write(f"=====================\n")
            f.write(f"Both correct: {n11}\n")
            f.write(f"M2 correct, M4 wrong: {n10}\n")
            f.write(f"M2 wrong, M4 correct: {n01}\n")
            f.write(f"Both wrong: {n00}\n")
            f.write(f"p-value: {p:.4f}\n")
    except Exception as e:
        print(f"  Error calculando McNemar: {e}")

    print("\n" + "=" * 60)
    print("  REPORTE GENERADO")
    print("=" * 60)
    print(f"Revisa la carpeta: {RESULTS_DIR.resolve()}")


if __name__ == '__main__':
    main()