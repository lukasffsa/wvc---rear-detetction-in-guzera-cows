import matplotlib.pyplot as plt
import numpy as np

def plot_fold_metrics(metrics_history):
    folds = list(metrics_history.keys())
    
    precisions = [metrics_history[f]['precision'] for f in folds]
    recalls = [metrics_history[f]['recall'] for f in folds]
    f_scores = [metrics_history[f]['f_score'] for f in folds]
    map75s = [metrics_history[f]['map75'] for f in folds]

    x = np.arange(len(folds))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - 1.5*width, precisions, width, label='Precision', color='#1f77b4')
    ax.bar(x - 0.5*width, recalls, width, label='Recall', color='#ff7f0e')
    ax.bar(x + 0.5*width, f_scores, width, label='F-Score', color='#2ca02c')
    ax.bar(x + 1.5*width, map75s, width, label='mAP@0.75', color='#d62728')

    ax.set_ylabel('Scores (0 a 1)')
    ax.set_title('Desempenho do Modelo de Detecção por Fold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('metricas_finais.png')