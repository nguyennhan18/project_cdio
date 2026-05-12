import cv2
import os
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Khởi tạo Global
model = YOLO('weights/best.pt')
LINE_Y = 400 # Vị trí vạch mặc định
CURRENT_VIDEO = None
total_counts = {"chicken": 0, "duck": 0}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global CURRENT_VIDEO, total_counts
    if 'video' not in request.files:
        return "No video part", 400
    file = request.files['video']
    if file.filename == '':
        return "No selected file", 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    CURRENT_VIDEO = filepath
    total_counts = {"chicken": 0, "duck": 0} # Reset đếm khi có video mới
    return jsonify({"status": "success", "filename": filename})

@app.route('/update_line', methods=['POST'])
def update_line():
    global LINE_Y
    data = request.json
    LINE_Y = int(data.get('y', 400))
    return jsonify({"status": "success", "new_y": LINE_Y})

def generate_frames():
    global CURRENT_VIDEO, LINE_Y, total_counts
    if not CURRENT_VIDEO:
        return
    
    cap = cv2.VideoCapture(CURRENT_VIDEO)
    counted_ids = set()
    prev_positions = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # AI Processing
        results = model.track(frame, persist=True, conf=0.5, tracker="bytetrack.yaml", verbose=False)
        
        # Vẽ các khung nhận diện (Boxes) lên frame
        frame = results[0].plot()
        
        # Vẽ vạch đếm
        h, w = frame.shape[:2]
        cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (255, 100, 0), 4)
        cv2.putText(frame, "DETECTION LINE", (10, LINE_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, obj_id, cls in zip(boxes, ids, clss):
                label = model.names[cls]
                cy = (box[1] + box[3]) // 2
                
                if obj_id in prev_positions:
                    if (prev_positions[obj_id] < LINE_Y <= cy) or (cy < LINE_Y <= prev_positions[obj_id]):
                        if obj_id not in counted_ids:
                            total_counts[label] += 1
                            counted_ids.add(obj_id)
                prev_positions[obj_id] = cy

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts')
def get_counts():
    return jsonify(total_counts)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
