import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison(weighted_metrics, bs):
    models = list(weighted_metrics.keys())
    
    precisions = [weighted_metrics[m]['precision'] for m in models]
    recalls = [weighted_metrics[m]['recall'] for m in models]
    f_scores = [weighted_metrics[m]['f_score'] for m in models]
    map50s = [weighted_metrics[m]['map50'] for m in models]
    map50_95s = [weighted_metrics[m]['map50_95'] for m in models]

    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - 2*width, precisions, width, label='Precision', color='#1f77b4')
    ax.bar(x - width, recalls, width, label='Recall', color='#ff7f0e')
    ax.bar(x, f_scores, width, label='F-Score', color='#2ca02c')
    ax.bar(x + width, map50s, width, label='mAP@0.5', color='#d62728')
    ax.bar(x + 2*width, map50_95s, width, label='mAP@0.5:0.95', color='#9467bd')

    ax.set_ylabel('Pontuação Média Ponderada')
    ax.set_title(f'Comparação de Modelos (Batch Size = {bs})')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    # Salva o gráfico distinguindo pelo batch size
    plt.savefig(f'metricas_comparativas_bs{bs}.png')
    plt.close()