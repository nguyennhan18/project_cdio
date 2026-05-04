# 🐔🦆 Hệ thống nhận diện và đếm Gà & Vịt
> Đồ án môn học — YOLOv8 + OpenCV + ByteTrack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Giới thiệu

Hệ thống sử dụng **YOLOv8** để nhận diện và phân biệt **gà** và **vịt** trong video/camera real-time, kết hợp **ByteTrack** để đếm số lượng chính xác mà không bị trùng.

| Class | Nhãn | Màu box |
|-------|------|---------|
| 0     | Gà   | 🟠 Cam  |
| 1     | Vịt  | 🔵 Xanh |

---

## 👥 Nhóm thực hiện

| Thành viên | Vai trò | Nhiệm vụ |
|------------|---------|----------|
| A | Data Lead | Thu thập, label, augment dataset |
| B | Model Lead | Train YOLOv8, đánh giá model |
| C | App Lead | Detect real-time, tích hợp đếm |
| D | PM + Docs | GitHub, báo cáo, slide |

---

## 📁 Cấu trúc dự án

```
ga-vit-detection/
├── notebooks/
│   └── train_colab.ipynb     ← Notebook train trên Google Colab
├── src/
│   ├── augment.py            ← Tăng cường dataset
│   ├── train.py              ← Train YOLOv8 (local)
│   ├── detect.py             ← Nhận diện real-time
│   └── check_model.py        ← Đánh giá model
├── dataset/                  ← KHÔNG push lên GitHub (xem hướng dẫn)
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   └── labels/val/
├── weights/                  ← best.pt sau khi train (share qua GDrive)
├── results/                  ← Confusion matrix, loss curve
├── data.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Hướng dẫn chạy nhanh

### 1. Clone repo
```bash
git clone https://github.com/YOUR_USERNAME/ga-vit-detection.git
cd ga-vit-detection
```

### 2. Cài thư viện
```bash
pip install -r requirements.txt
```

### 3. Tải dataset
- **Gà:** https://universe.roboflow.com/chicken-detection-e7acb/chicken-detection-gkoje
- **Vịt:** https://universe.roboflow.com/duck-count-detection/duck-detection-and-counting

Sau khi tải, đặt vào thư mục `dataset/` theo đúng cấu trúc trên.

### 4. Train trên Google Colab
Mở file `notebooks/train_colab.ipynb` → chạy từng cell theo thứ tự.

### 5. Nhận diện real-time
```bash
# Webcam
python src/detect.py --source 0

# Video file
python src/detect.py --source video.mp4 --save
```

---

## 📊 Kết quả

| Metric | Giá trị |
|--------|---------|
| mAP50  | > 0.85  |
| FPS    | ~25–30  |
| Classes| 2 (gà, vịt) |

> Cập nhật sau khi train xong.

---

## 🔗 Tài nguyên

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
- [Roboflow Universe](https://universe.roboflow.com)
- [Google Colab Notebook](notebooks/train_colab.ipynb)
- [Model weights (Google Drive)](https://drive.google.com/YOUR_LINK) ← cập nhật sau khi train

---

## 📝 License
MIT License — tự do sử dụng cho mục đích học tập.
