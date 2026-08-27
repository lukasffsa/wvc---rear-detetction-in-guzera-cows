from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import torch

def FasterRCNN(num_classes=2):

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    num_classes = 2

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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


# Função para avaliar acurácia no conjunto de validação
def evaluate_metrics(model, data_loader, device, score_threshold=0.5):
    model.eval()
    
    # Contadores para IoU >= 0.5
    TP_50, FP_50, FN_50 = 0, 0, 0
    # Contadores para IoU >= 0.75
    TP_75, FP_75 = 0, 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                gt_boxes = target['boxes']
                keep = pred['scores'] >= score_threshold
                pred_boxes = pred['boxes'][keep]

                # Imagem de Background (Sem boi)
                if len(gt_boxes) == 0:
                    if len(pred_boxes) > 0:
                        FP_50 += 1
                        FP_75 += 1
                    continue

                # Havia boi, mas o modelo não previu nada
                if len(pred_boxes) == 0:
                    FN_50 += 1
                    continue

                # Pega a previsão com maior confiança
                best_pred_box = pred_boxes[0].cpu().numpy()
                gt_box = gt_boxes[0].cpu().numpy()
                iou = compute_iou(best_pred_box, gt_box)

                # Avaliação em IoU 0.5 (Para Precision, Recall, F-Score e mAP50)
                if iou >= 0.5:
                    TP_50 += 1
                else:
                    FP_50 += 1
                    FN_50 += 1 # Conta como FN pois o boi real não foi detectado corretamente
                    
                # Avaliação restrita em IoU 0.75 (Para mAP75)
                if iou >= 0.75:
                    TP_75 += 1
                else:
                    FP_75 += 1

    # Cálculos matemáticos
    precision = TP_50 / (TP_50 + FP_50) if (TP_50 + FP_50) > 0 else 0.0
    recall = TP_50 / (TP_50 + FN_50) if (TP_50 + FN_50) > 0 else 0.0
    
    f_score = 0.0
    if (precision + recall) > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
        
    map75 = TP_75 / (TP_75 + FP_75) if (TP_75 + FP_75) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "map50": precision, # Neste cenário de 1 classe, Precision aproxima o AP@0.5
        "map75": map75
    }