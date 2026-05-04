"""
detect.py — Nhận diện & đếm Gà/Vịt real-time từ webcam hoặc video
Yêu cầu: pip install ultralytics opencv-python

Cách dùng:
  python src/detect.py --source 0              # webcam
  python src/detect.py --source video.mp4      # video file
  python src/detect.py --source video.mp4 --save  # lưu kết quả
"""

import cv2
import argparse
import torch
from ultralytics import YOLO
from collections import defaultdict

CLASS_COLORS = {
    "ga":  (0,  140, 255),   # cam — gà
    "vit": (255, 144, 30),   # xanh — vịt
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=0,
                   help="0=webcam | đường dẫn video")
    p.add_argument("--model", default="weights/best.pt",
                   help="Đường dẫn model .pt")
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--iou",  type=float, default=0.50)
    p.add_argument("--save", action="store_true", default=False)
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = YOLO(args.model)
    names = model.names

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Không mở được: {source}"); return

    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter("output_ga_vit.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("Nhấn Q để thoát.")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        results = model.predict(frame, conf=args.conf, iou=args.iou,
                                verbose=False, device=device)[0]

        counts = defaultdict(int)
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = names[cls_id]
            color  = CLASS_COLORS.get(label, (200, 200, 200))
            counts[label] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(frame, text, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        # Bảng đếm
        lines = [
            f"Frame: {frame_idx}",
            f"Ga:    {counts['ga']}",
            f"Vit:   {counts['vit']}",
            f"Tong:  {counts['ga'] + counts['vit']}",
        ]
        overlay = frame.copy()
        cv2.rectangle(overlay, (12,12), (175, 12 + len(lines)*24 + 12), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, line in enumerate(lines):
            color = (255,255,255)
            if "Ga"  in line: color = CLASS_COLORS["ga"]
            if "Vit" in line: color = CLASS_COLORS["vit"]
            cv2.putText(frame, line, (20, 32 + i*24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        if writer: writer.write(frame)
        cv2.imshow("Ga & Vit Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print("Đã thoát.")

if __name__ == "__main__":
    main()
