import cv2
import torch
from ultralytics import YOLO
import argparse

# Parse command-line arguments for device selection
parser = argparse.ArgumentParser(description='YOLOv11s Emotion Detection Demo')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                    help='Device to run the model on: "cpu" or "cuda" (default: cpu)')
args = parser.parse_args()

# Define paths
VIDEO_PATH = r"D:\College\Semester 8\Skripsi\Program\5. Demo App\Input\demo_video.mp4"
YOLO_MODEL_PATH = r"D:\College\Semester 8\Skripsi\Program\Models\YOLOv11s_Emotion_Detection.pt"
OUTPUT_VIDEO_PATH = r"D:\College\Semester 8\Skripsi\Program\5. Demo App\Output\demo_video_ss_output.mp4"

# Emotion labels and corresponding colors (BGR format for OpenCV)
emotion_info = {
    'Angry': {'color': (0, 0, 255)},    # Red
    'Disgust': {'color': (0, 165, 255)}, # Orange
    'Fear': {'color': (0, 255, 255)},   # Yellow
    'Happy': {'color': (0, 255, 0)},    # Green
    'Neutral': {'color': (255, 0, 0)},  # Blue
    'Sad': {'color': (128, 0, 128)},    # Purple
    'Surprise': {'color': (203, 192, 255)} # Pink
}

# Set device
if args.device == 'cuda' and not torch.cuda.is_available():
    print("GPU is not available, falling back to CPU.")
    device = torch.device('cpu')
else:
    device = torch.device(args.device)
print(f"Using device: {device}")

# Load YOLOv11s model for emotion detection
yolo_model = YOLO(YOLO_MODEL_PATH)

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define display size
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Initialize video writer (original resolution)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
# Optional: Save output video at 1366x768
# out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Emotion detection with YOLOv11s
    results = yolo_model(frame, device=device)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    classes = results[0].boxes.cls.cpu().numpy()  # Class indices

    # Calculate scaling factors for display
    scale_x = DISPLAY_WIDTH / frame_width
    scale_y = DISPLAY_HEIGHT / frame_height

    # Store bounding box info
    box_info = []

    for box, conf, cls in zip(boxes, confidences, classes):
        if conf > 0.25:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)
            emotion = list(emotion_info.keys())[int(cls)]  # Map class index to emotion
            box_info.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'emotion': emotion, 'confidence': conf,
                'color': emotion_info[emotion]['color']
            })

    # Draw bounding boxes and labels on original frame (for saving)
    for info in box_info:
        x1, y1, x2, y2 = info['x1'], info['y1'], info['x2'], info['y2']
        box_color = info['color']
        label = f"{info['emotion']}: {info['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    # Resize frame for display
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)

    # Draw bounding boxes and labels on resized display frame
    for info in box_info:
        x1_display = int(info['x1'] * scale_x)
        y1_display = int(info['y1'] * scale_y)
        x2_display = int(info['x2'] * scale_x)
        y2_display = int(info['y2'] * scale_y)
        box_color = info['color']
        label = f"{info['emotion']}: {info['confidence']:.2f}"
        cv2.rectangle(display_frame, (x1_display, y1_display), (x2_display, y2_display), box_color, 2)
        cv2.putText(display_frame, label, (x1_display, y1_display - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    # Display the resized frame
    cv2.imshow('YOLOv11s Emotion Detection', display_frame)
    out.write(frame)  # Save original frame
    # Optional: Save resized frame
    # out.write(display_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved at: {OUTPUT_VIDEO_PATH}")