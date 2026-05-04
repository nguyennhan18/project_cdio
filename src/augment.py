"""
augment.py — Tăng cường dataset gà & vịt bằng Albumentations
Dùng khi chạy local, nếu dùng Colab thì augment đã tích hợp trong notebook

Cách dùng: python src/augment.py
"""

import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A

IMG_DIR       = Path("dataset/images/train")
LABEL_DIR     = Path("dataset/labels/train")
AUG_PER_IMAGE = 4
IMG_SIZE      = 640

random.seed(42); np.random.seed(42)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.15),
    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.25,
                       rotate_limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
    A.Perspective(scale=(0.03, 0.08), p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
        A.RandomGamma(gamma_limit=(70, 140)),
        A.CLAHE(clip_limit=4.0),
    ], p=0.7),
    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=30),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1),
    ], p=0.6),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3,5)),
        A.MotionBlur(blur_limit=5),
    ], p=0.25),
    A.GaussNoise(var_limit=(10,50), p=0.2),
    A.RandomShadow(p=0.2),
    A.CoarseDropout(max_holes=4, max_height=40, max_width=40,
                    min_holes=1, min_height=10, min_width=10, p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'],
                             min_visibility=0.3, min_area=100))

originals = [p for p in IMG_DIR.glob('*')
             if p.suffix.lower() in {'.jpg','.jpeg','.png'} and '_aug' not in p.stem]

print(f"{len(originals)} ảnh gốc → tạo thêm {len(originals)*AUG_PER_IMAGE}")

created = skipped = 0
for img_path in tqdm(originals):
    lbl_path = LABEL_DIR / (img_path.stem + '.txt')
    if not lbl_path.exists(): skipped += 1; continue

    img = cv2.imread(str(img_path))
    if img is None: skipped += 1; continue
    img = cv2.cvtColor(cv2.resize(img,(IMG_SIZE,IMG_SIZE)), cv2.COLOR_BGR2RGB)

    cids, bboxes = [], []
    with open(lbl_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p)==5:
                cids.append(int(p[0]))
                bboxes.append([max(0.001,min(0.999,float(x))) for x in p[1:]])

    if not bboxes: skipped += 1; continue

    for idx in range(AUG_PER_IMAGE):
        try:
            out = transform(image=img, bboxes=bboxes, class_labels=cids)
            if not out['bboxes']: continue
            stem = f"{img_path.stem}_aug{idx:02d}"
            cv2.imwrite(str(IMG_DIR/(stem+'.jpg')),
                        cv2.cvtColor(out['image'],cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(LABEL_DIR/(stem+'.txt'),'w') as f:
                for cls,(x,y,w,h) in zip(out['class_labels'],out['bboxes']):
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            created += 1
        except: pass

print(f"\nTạo: {created} | Bỏ qua: {skipped}")
print(f"Tổng train sau augment: {len(list(IMG_DIR.glob('*')))} ảnh")
