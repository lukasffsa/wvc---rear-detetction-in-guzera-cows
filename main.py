import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader.Dataset import CustomDataset, transform
from models.SSDLite320_MobileNetV3_Large import SSDLiteModel, evaluate_metrics as evaluate_ssd_metrics
from models.FRCNN_resnet50 import FasterRCNN, evaluate_metrics
from views.views import visualize_predictions
from PIL import Image
from charts.metrics_charts import plot_fold_metrics

def main():
 
    os.makedirs("weights", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando na {device}")

    # Configurações do Dataset
    root_dir = 'dataset\\dataset_detection'
    all_folds = [1, 2, 3, 4, 5]
    num_epochs = 20

    fold_accs = []
    metrics_history = {}

    # Loop de 5-folds
    for fold in all_folds:
        print(f"\n{'='*40}")
        print(f"Iniciando Fold de Validação: {fold}")

        # Divisão 4/1
        train_fold_names = [f'fold_{i}' for i in all_folds if i != fold]
        val_fold_names = [f'fold_{fold}']

        train_dataset = CustomDataset(root_dir, train_fold_names, transforms=transform)
        valid_dataset = CustomDataset(root_dir, val_fold_names, transforms=transform)

        train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, drop_last=True, collate_fn=lambda x: tuple(zip(*x)))
        valid_loader = DataLoader(valid_dataset, batch_size=5, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

        # Inicializa o modelo
        model = SSDLiteModel(num_classes=2)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # Loop de treinamento SEM early stopping
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for images, targets in train_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()
                optimizer.step()
                train_loss += losses.item()

            lr_scheduler.step()
            
            print(f'Fold {fold} | Epoch: {epoch + 1}/{num_epochs} | Loss: {train_loss / len(train_loader):.4f}')

        # 4. Avaliação completa da última época
        print(f"Avaliando o modelo da última época do Fold {fold}...")
        metrics = evaluate_ssd_metrics(model, valid_loader, device, score_threshold=0.5)
        
        print(f"Resultados Fold {fold}:")
        print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F-Score: {metrics['f_score']:.4f}")
        print(f"mAP@0.5: {metrics['map50']:.4f} | mAP@0.75: {metrics['map75']:.4f}")
        
        metrics_history[fold] = metrics
        
        # Guardando o mAP50 para calcular a média e desvio padrão finais
        fold_accs.append(metrics['map50'])

        # --- BLOCO PARA SALVAR OS PESOS ---
        caminho_pesos = f"weights/modelo_fold_{fold}.pth"
        torch.save(model.state_dict(), caminho_pesos)
        print(f"Pesos do Fold {fold} salvos com sucesso em: {caminho_pesos}")
        # ----------------------------------

        # --- BLOCO DE VISUALIZAÇÃO ---
        print("Gerando visualização para uma imagem deste fold...")
        new_image_path, _ = valid_dataset.samples[0]
        
        img_pil = Image.open(new_image_path).convert("RGB")
        img_tensor = transform(img_pil).to(device)
        
        model.eval()
        with torch.no_grad():
            new_prediction = model([img_tensor])

        _, ground_truth_target = valid_dataset[0]

        visualize_predictions(fold, new_image_path, new_prediction[0], ground_truth_target=ground_truth_target, score_threshold=0.5)
        # -----------------------------

    # Resultados de todos os folds
    print("\nGerando gráfico de métricas...")
    plot_fold_metrics(metrics_history)

    # Cálculo da média e desvio padrão finais (Foco no mAP@0.5)
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    print("\n" + "="*40)
    print("Resultados Finais 5-Fold CV:")
    print(f"mAP@0.5 Médio: {mean_acc:.4f} ± {std_acc:.4f}")

if __name__ == '__main__':
    main()