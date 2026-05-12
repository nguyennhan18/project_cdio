
import logging
from ultralytics import YOLO
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model_path="weights/best.pt", data_yaml="dataset/data.yaml"):
    if not Path(model_path).exists():
        logger.error("Không tìm thấy model!")
        return

    model = YOLO(model_path)
    logger.info("Đang chạy Validation...")
    
    metrics = model.val(data=data_yaml)
    
    print("\n" + "="*40)
    print(f"BÁO CÁO KẾT QUẢ CHO: {model_path}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_model()
