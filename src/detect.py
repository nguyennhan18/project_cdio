import cv2
import argparse
from ultralytics import YOLO

def simple_detect(source_path):
    # 1. Thiết lập vạch đếm (Vị trí Y)
    LINE_Y = 450 # Bạn có thể điều chỉnh độ cao của vạch này (từ 0 đến chiều cao video)
    
    # 2. Khởi tạo
    model = YOLO('weights/best.pt')
    counted_ids = set()
    total_counts = {"chicken": 0, "duck": 0}
    prev_positions = {} # Lưu vị trí Y của khung hình trước đó

    # 3. Chạy nhận diện
    results = model.track(source=source_path, stream=True, conf=0.5, persist=True, tracker="bytetrack.yaml")

    print(f"--- Đang chạy chế độ ĐẾM QUA VẠCH (Line: {LINE_Y}) ---")
    
    for r in results:
        frame = r.plot()
        w, h = frame.shape[1], frame.shape[0]
        
        # Vẽ vạch đếm (Màu xanh dương)
        cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (255, 0, 0), 5)
        cv2.putText(frame, "VACH DEM", (20, LINE_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            ids = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for box, obj_id, cls in zip(boxes, ids, clss):
                label = model.names[cls]
                cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                
                # Logic đếm qua vạch
                if obj_id in prev_positions:
                    prev_y = prev_positions[obj_id]
                    # Nếu vật thể đi từ trên xuống dưới vạch HOẶC từ dưới lên trên vạch
                    if (prev_y < LINE_Y <= cy) or (cy < LINE_Y <= prev_y):
                        if obj_id not in counted_ids:
                            total_counts[label] += 1
                            counted_ids.add(obj_id)
                            print(f"🎯 CẮT VẠCH: {label} (ID:{obj_id}) | Tổng: {total_counts}")
                
                # Cập nhật vị trí Y hiện tại cho ID này
                prev_positions[obj_id] = cy

        # Hiển thị Dashboard
        cv2.rectangle(frame, (10, 10), (250, 90), (0, 0, 0), -1) # Nền đen cho bảng điểm
        cv2.putText(frame, f"CHICKEN: {total_counts['chicken']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"DUCK: {total_counts['duck']}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Poultry Line Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"--- Kết quả cuối cùng: {total_counts} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    VIDEO_PATH = "/Users/nguyennhan18/Documents/Personal_documents/Project/yolo_env/nhom_ML/demvo.mp4" 
    parser.add_argument("--source", type=str, default=VIDEO_PATH)
    args = parser.parse_args()
    simple_detect(args.source)
