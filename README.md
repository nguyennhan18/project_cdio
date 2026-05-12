# 🐔🦆 Hệ thống AI Nhận diện & Đếm Gà/Vịt Chuyên Nghiệp
> **Giải pháp thị giác máy tính:** YOLOv8 + ByteTrack + Albumentations

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Giới thiệu

Dự án này là một hệ thống thị giác máy tính toàn diện nhằm nhận diện và đếm số lượng gà, vịt trong video hoặc camera thời gian thực. Khác với các hệ thống đếm thông thường, dự án này tích hợp **ByteTrack** để quản lý danh tính (ID) vật thể, giúp đếm chính xác số lượng cá thể duy nhất mà không bị trùng lặp khi chúng di chuyển hoặc bị che khuất tạm thời.

### ✨ Tính năng nổi bật
- **Real-time Tracking:** Sử dụng ByteTrack để duy trì ID cho từng cá thể.
- **Unique Counting:** Đếm tổng số lượng dựa trên tập hợp ID duy nhất đã xuất hiện.
- **Professional UI:** Giao diện dashboard hiển thị thống kê chuyên nghiệp trực tiếp trên frame.
- **Robust Augmentation:** Pipeline tăng cường dữ liệu mạnh mẽ với Albumentations.
- **Deep Evaluation:** Công cụ đánh giá chuyên sâu, tự động trích xuất các trường hợp model dự đoán sai để phân tích.

---

## 📁 Cấu trúc dự án (OOP Design)

```text
project_cdio/
├── src/
│   ├── detect.py         ← 🚀 Module chính: Nhận diện & Tracking (Class Detector)
│   ├── augment.py        ← 🛠️ Module tăng cường dữ liệu (Class DataAugmenter)
│   └── check_model.py    ← 📊 Module đánh giá chuyên sâu (Class ModelEvaluator)
├── dataset/              ← Dữ liệu huấn luyện (YOLO format)
│   └── data.yaml         ← Cấu hình thông tin class
├── weights/              ← Nơi lưu trữ model checkpoint (*.pt)
├── evaluation_results/   ← Kết quả đánh giá và ảnh dự đoán sai
├── requirements.txt      ← Các thư viện cần thiết
└── README.md
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Tăng cường dữ liệu (Data Augmentation)
Nếu bạn có ít ảnh gốc, hãy chạy script sau để tạo thêm các biến thể:
```bash
python src/augment.py
```

### 3. Đánh giá Model
Sau khi huấn luyện, hãy kiểm tra các chỉ số (mAP, Precision, Recall) và xem các ảnh model dự đoán sai:
```bash
python src/check_model.py
```

### 4. Chạy hệ thống Real-time
Hệ thống hỗ trợ cả Webcam và Video file:
```bash
# Chạy với Webcam
python src/detect.py --source 0

# Chạy với Video và lưu kết quả
python src/detect.py --source data/test_video.mp4 --save
```

---

## 📊 Kết quả đạt được

| Chỉ số | Giá trị |
|--------|---------|
| mAP50  | > 0.85  |
| FPS    | ~30 (trên GPU) |
| Tracker| ByteTrack |

---

## 🔗 Liên hệ & Đóng góp
Dự án được thực hiện nhằm mục đích nghiên cứu và học tập. Mọi đóng góp vui lòng tạo Issue hoặc Pull Request.

---
**License:** MIT
