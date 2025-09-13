import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
from transformers import ViTModel, ViTConfig
import argparse

# Parse command-line arguments for device selection
parser = argparse.ArgumentParser(description='Facial Expression Recognition Demo')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                    help='Device to run the model on: "cpu" or "cuda" (default: cpu)')
args = parser.parse_args()

# Define paths
VIDEO_PATH = r"D:\College\Semester 8\Skripsi\Program\5. Demo App\Input\demo_video.mp4"
YOLO_MODEL_PATH = r"D:\College\Semester 8\Skripsi\Program\Models\YOLOv11s_Face_Detection.pt"
HYBRIDVIT_MODEL_PATH = r"D:\College\Semester 8\Skripsi\Program\Models\HybridViT_ResNet-50_FER.pth"
OUTPUT_VIDEO_PATH = r"D:\College\Semester 8\Skripsi\Program\5. Demo App\Output\demo_video_ds_output.mp4"

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

# Load YOLOv11s model for face detection
yolo_model = YOLO(YOLO_MODEL_PATH)

# Define HybridViT model
class HybridViT(nn.Module):
    def __init__(self, num_classes=7, embed_dim=768):
        super(HybridViT, self).__init__()

        # ResNet-50 Backbone
        resnet = models.resnet50(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC and avgpool

        # Adjust channels and upsample to match ViT input size
        self.conv = nn.Conv2d(2048, 3, kernel_size=1)  # Match ViT's expected 3 channels
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        # ViT Configuration
        vit_config = ViTConfig(
            hidden_size=embed_dim,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_channels=3  # Standard RGB input
        )
        self.vit = ViTModel(vit_config)

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Extract features with ResNet-50
        features = self.resnet_features(x)  # Shape: (batch, 2048, 7, 7)
        features = self.conv(features)  # Shape: (batch, 3, 7, 7)
        features = self.upsample(features)  # Shape: (batch, 3, 224, 224)

        # Pass through ViT
        outputs = self.vit(pixel_values=features)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token

        # Classification
        logits = self.classifier(pooled_output)
        return logits

# Load HybridViT model
hybridvit_model = HybridViT(num_classes=len(emotion_info)).to(device)
hybridvit_model.load_state_dict(torch.load(HYBRIDVIT_MODEL_PATH, map_location=device))
hybridvit_model.eval()

# Image preprocessing for HybridViT
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

    # Face detection with YOLOv11s
    results = yolo_model(frame, device=device)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Calculate scaling factors for display
    scale_x = DISPLAY_WIDTH / frame_width
    scale_y = DISPLAY_HEIGHT / frame_height

    # Store bounding box info (coordinates, emotion, confidence)
    box_info = []

    for box, conf in zip(boxes, confidences):
        if conf > 0.25:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)
            # Extract face ROI
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Preprocess face for HybridViT
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_tensor = preprocess(face_rgb).unsqueeze(0).to(device)

            # Facial expression recognition
            with torch.no_grad():
                outputs = hybridvit_model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                emotion = list(emotion_info.keys())[predicted.item()]
                emotion_conf = confidence.item()

            # Store box info
            box_info.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'emotion': emotion, 'confidence': emotion_conf,
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
    cv2.imshow('Facial Expression Recognition', display_frame)
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