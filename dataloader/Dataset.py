from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import torch

class CustomDataset(Dataset):
    def __init__(self, root_dir, fold_names, transforms=None):
        self.transforms = transforms
        self.samples = []  # lista de tuplas (caminho_img, caminho_label)

        for fold in fold_names:
            images_dir = os.path.join(root_dir, fold, 'images')
            labels_dir = os.path.join(root_dir, fold, 'labels')

            img_files = sorted([f for f in os.listdir(images_dir)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

            for img_name in img_files:
                img_path = os.path.join(images_dir, img_name)
                label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')

                self.samples.append((img_path, label_path))

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        boxes, labels = [], []
        
        # Só tenta ler o arquivo se ele existir e não for vazio
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    class_id, x_c, y_c, w, h = map(float, line.split())
                    x1 = (x_c - w / 2) * img_width
                    y1 = (y_c - h / 2) * img_height
                    x2 = (x_c + w / 2) * img_width
                    y2 = (y_c + h / 2) * img_height
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)  # única classe (bovino)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.samples)

# Apenas exportamos o transform (o DataLoader agora fica no main.py)
transform = T.Compose([T.ToTensor()])