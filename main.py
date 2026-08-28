import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader.Dataset import CustomDataset, transform
from models.SSDLite320_MobileNetV3_Large import SSDLiteModel, evaluate_metrics as evaluate_ssd_metrics
from models.FRCNN_resnet50 import FasterRCNN, evaluate_metrics as evaluate_frcnn_metrics
from views.views import visualize_predictions
from PIL import Image
from charts.metrics_charts import plot_model_comparison

def main():
    os.makedirs("weights", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando na {device}")

    root_dir = 'dataset\\dataset_detection'
    all_folds = [1, 2, 3, 4, 5]
    num_epochs = 20
    
    batch_sizes = [2, 4, 6]
    
    models_dict = {
        "FasterRCNN": (FasterRCNN, evaluate_frcnn_metrics),
        "SSDLite": (SSDLiteModel, evaluate_ssd_metrics)
    }

    # Loop 1: Itera pelos tamanhos de lote (Batch Sizes)
    for bs in batch_sizes:
        print(f"\n{'#'*50}\nIniciando experimentos com Batch Size: {bs}\n{'#'*50}")
        weighted_metrics_per_model = {}

        # Loop 2: Itera pelos dois modelos
        for model_name, (ModelClass, eval_func) in models_dict.items():
            print(f"\n--- Treinando Modelo: {model_name} (BS={bs}) ---")
            
            # Inicializa acumuladores ponderados zerados
            weighted_metrics = {"precision": 0.0, "recall": 0.0, "f_score": 0.0, "map50": 0.0, "map50_95": 0.0}

            # Loop 3: Validação Cruzada (5-Fold)
            for fold in all_folds:
                print(f"\nIniciando Fold: {fold}")
                train_fold_names = [f'fold_{i}' for i in all_folds if i != fold]
                val_fold_names = [f'fold_{fold}']

                train_dataset = CustomDataset(root_dir, train_fold_names, transforms=transform)
                valid_dataset = CustomDataset(root_dir, val_fold_names, transforms=transform)

                # Dataloaders usando o Batch Size atual
                train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True, collate_fn=lambda x: tuple(zip(*x)))
                valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

                # Cálculo do peso do fold (Imagens do fold / Imagens totais)
                total_images = len(train_dataset) + len(valid_dataset)
                fold_weight = len(valid_dataset) / total_images

                model = ModelClass(num_classes=2)
                model.to(device)

                params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

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
                    print(f'{model_name} | Fold {fold} | Epoca: {epoch + 1}/{num_epochs} | Loss: {train_loss / len(train_loader):.4f}')

                # Avaliação na última época usando as métricas com IoU atualizado
                metrics = eval_func(model, valid_loader, device, score_threshold=0.5)
                
                # Multiplica pela quantidade de imagens e soma na métrica final do modelo
                for key in weighted_metrics.keys():
                    weighted_metrics[key] += metrics[key] * fold_weight

                # Salva pesos com nomenclatura para evitar sobreposição
                caminho_pesos = f"weights/{model_name}_bs{bs}_fold{fold}.pth"
                torch.save(model.state_dict(), caminho_pesos)

                # Geração de Visualização
                new_image_path, _ = valid_dataset.samples[0]
                img_pil = Image.open(new_image_path).convert("RGB")
                img_tensor = transform(img_pil).to(device)
                
                model.eval()
                with torch.no_grad():
                    new_prediction = model([img_tensor])

                _, ground_truth_target = valid_dataset[0]
                
                # O nome do arquivo gerado usará a string para separar as visualizações
                identifier = f"{model_name}_bs{bs}_fold{fold}"
                visualize_predictions(identifier, new_image_path, new_prediction[0], ground_truth_target=ground_truth_target, score_threshold=0.5)

            # Salva o resultado final ponderado do modelo atual
            weighted_metrics_per_model[model_name] = weighted_metrics
            
            print(f"\nResultados Finais Ponderados ({model_name} - BS {bs}):")
            for k, v in weighted_metrics.items():
                print(f"{k}: {v:.4f}")

        # Final do ciclo do Batch Size: Gera o gráfico comparativo
        plot_model_comparison(weighted_metrics_per_model, bs)

if __name__ == '__main__':
    main()