"""
check_model.py — Đánh giá model sau khi train
Xem confusion matrix, tìm ảnh dự đoán sai để biết cần bổ sung data gì

Cách dùng: python src/check_model.py
"""

from ultralytics import YOLO
import os, shutil
from pathlib import Path

MODEL_PATH = "weights/best.pt"
DATA_YAML  = "data.yaml"

model = YOLO(MODEL_PATH)

print("=== Đánh giá model ===")
metrics = model.val(data=DATA_YAML, conf=0.45, iou=0.50, plots=True)

print(f"\nmAP50:          {metrics.box.map50:.3f}  (mục tiêu > 0.85)")
print(f"mAP50-95:       {metrics.box.map:.3f}")
print(f"Precision Gà:   {metrics.box.p[0]:.3f}")
print(f"Precision Vịt:  {metrics.box.p[1]:.3f}")
print(f"Recall Gà:      {metrics.box.r[0]:.3f}")
print(f"Recall Vịt:     {metrics.box.r[1]:.3f}")

print("\n=== Hướng dẫn đọc Confusion Matrix ===")
print("Xem file: runs/val/confusion_matrix_normalized.png")
print("  Nếu [gà→vịt] cao  → thêm ảnh gà đa dạng hơn")
print("  Nếu [vịt→gà] cao  → thêm ảnh vịt với mỏ dẹt rõ hơn")
print("  Nếu [bg] cao      → giảm conf threshold hoặc thêm ảnh xa")

print("\n=== Lưu ảnh sai để review ===")
save_dir = Path("review_wrong")
save_dir.mkdir(exist_ok=True)

val_img_dir = Path("dataset/images/val")
wrong = 0
for img in list(val_img_dir.glob("*.jpg"))[:200]:
    res = model.predict(str(img), conf=0.45, verbose=False)[0]
    lbl = Path("dataset/labels/val") / (img.stem + ".txt")
    if not lbl.exists(): continue

    gt = set()
    with open(lbl) as f:
        for line in f:
            gt.add(int(line.split()[0]))

    pred = set()
    if res.boxes:
        for box in res.boxes:
            pred.add(int(box.cls[0]))

    if gt != pred:
        wrong += 1
        shutil.copy(img, save_dir / img.name)
        res.save(filename=str(save_dir / f"pred_{img.name}"))

print(f"Tìm thấy {wrong} ảnh sai → xem trong: review_wrong/")
