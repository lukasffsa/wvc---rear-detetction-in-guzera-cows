from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import torch
import contextlib
import io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def SSDLiteModel(num_classes=2):
    
    model = ssdlite320_mobilenet_v3_large(weights_backbone='DEFAULT', num_classes=num_classes)

    return model

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def evaluate_metrics(model, data_loader, device, score_threshold=0.5):
    model.eval()
    
    # 1. Estruturas em memória exigidas pelo pycocotools
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "boi"}] # Classe genérica
    }
    coco_predictions = []
    ann_id = 1
    
    # Contadores manuais mantidos para Precision, Recall e F-Score (compatibilidade com o gráfico)
    TP_50, FP_50, FN_50 = 0, 0, 0
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            images = list(img.to(device) for img in images)
            predictions = model(images)

            for img_t, target, pred in zip(images, targets, predictions):
                # Extrai um ID único para a imagem
                img_id = int(target["image_id"].item()) if "image_id" in target else idx * 1000 + ann_id
                
                coco_dataset["images"].append({
                    "id": img_id, 
                    "width": img_t.shape[2], 
                    "height": img_t.shape[1]
                })

                # --- Popula Ground Truth para o COCO ---
                gt_boxes = target['boxes'].cpu().numpy()
                for box in gt_boxes:
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    coco_dataset["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0
                    })
                    ann_id += 1

                # --- Popula Predições para o COCO ---
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                for box, score in zip(pred_boxes, pred_scores):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    coco_predictions.append({
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score)
                    })

                # --- Lógica Manual Estrita para Precision/Recall ---
                keep = pred['scores'] >= score_threshold
                filtered_pred_boxes = pred['boxes'][keep]

                if len(gt_boxes) == 0:
                    if len(filtered_pred_boxes) > 0:
                        FP_50 += 1
                    continue

                if len(filtered_pred_boxes) == 0:
                    FN_50 += 1
                    continue

                best_pred_box = filtered_pred_boxes[0].cpu().numpy()
                gt_box = gt_boxes[0]
                iou = compute_iou(best_pred_box, gt_box)

                if iou >= 0.5:
                    TP_50 += 1
                else:
                    FP_50 += 1
                    FN_50 += 1

    # Cálculo da Precision, Recall e F-Score do seu pipeline base
    precision = TP_50 / (TP_50 + FP_50) if (TP_50 + FP_50) > 0 else 0.0
    recall = TP_50 / (TP_50 + FN_50) if (TP_50 + FN_50) > 0 else 0.0
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # --- Execução Oficial do pycocotools ---
    coco_gt = COCO()
    coco_gt.dataset = coco_dataset
    
    map50, map50_95 = 0.0, 0.0
    
    # Usamos io.StringIO() para silenciar os prints gigantescos padrão da pycocotools
    with contextlib.redirect_stdout(io.StringIO()): 
        coco_gt.createIndex()
        
        if len(coco_predictions) > 0:
            coco_dt = coco_gt.loadRes(coco_predictions)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extraindo as métricas de mAP específicas
            map50_95 = coco_eval.stats[0] # Average Precision (AP) @[ IoU=0.50:0.95 ]
            map50 = coco_eval.stats[1]    # Average Precision (AP) @[ IoU=0.50 ]

    return {
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "map50": map50, 
        "map50_95": map50_95
    }