# OpenVision AI - Computer Vision Strategy

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OpenVision AI Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   Browser    │     │   Uploaded   │     │    Edge      │                 │
│  │ Live Stream  │     │   Video     │     │  Deployment  │                 │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘                 │
│         │                    │                    │                         │
│         ▼                    ▼                    ▼                         │
│  ┌──────────────────────────────────────────────────────────────┐            │
│  │                   Detection Layer                             │            │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│           │
│  │  │TensorFlow.js│  │   YOLOv8    │  │     MediaPipe           ││           │
│  │  │ coco-ssd   │  │ FastAPI     │  │ Face/Hand/Pose Landmarks│           │
│  │  │ (80 classes)│  │ (Ultralytics│  │                        ││           │
│  │  │ + MoveNet   │  │  Server)    │  │                        ││           │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘│           │
│  └──────────────────────────────────────────────────────────────┘            │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐            │
│  │                    Event Streaming Layer                       │            │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│           │
│  │  │  Kinesis    │  │  WebSocket  │  │       SNS               ││           │
│  │  │Data Streams │  │  (Live UI)  │  │    (Alerts)             ││           │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘│           │
│  └──────────────────────────────────────────────────────────────┘            │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐            │
│  │                    React Dashboard                            │            │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│           │
│  │  │ Real-time  │  │   Canvas    │  │      Rules              ││           │
│  │  │ Detection  │  │  Overlay    │  │     Engine              ││           │
│  │  │   Hooks     │  │  Rendering │  │                        ││           │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘│           │
│  └──────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Live Stream Processing (Browser-Based)

### Technology Stack

| Component | Library/Service | Purpose |
|-----------|-----------------|---------|
| Object Detection | `@tensorflow-models/coco-ssd` | Real-time detection (80 COCO classes) |
| Pose Detection | `@tensorflow-models/pose-detection` (MoveNet) | Movement & attention tracking |
| Landmarking | `@mediapipe/tasks-vision` | Face, hand, and pose landmarks |
| Camera Access | WebRTC `getUserMedia` | Live video stream from browser |
| Visualization | Canvas API + React | Bounding box overlay rendering |

### Implementation Details

#### 1.1 Object Detection (`coco-ssd`)

```typescript
// Real-time object detection hook
import * as cocoSsd from '@tensorflow-models/coco-ssd';

interface Detection {
  bbox: [number, number, number, number]; // [x, y, width, height]
  label: string;
  score: number;
}

async function useObjectDetection(videoRef: React.RefObject<HTMLVideoElement>) {
  const model = await cocoSsd.load();
  // ...
}
```

**Supported Classes**: Person, Phone, Laptop, Book, Cup, Chair, etc. (80 COCO categories)

#### 1.2 Pose Detection (MoveNet)

```typescript
import * as poseDetection from '@tensorflow-models/pose-detection';

interface Pose {
  keypoints: Array<{
    x: number;
    y: number;
    score: number;
    name: string; // nose, left_eye, right_eye, etc.
  }>;
  score: number;
}
```

**Keypoints Tracked** (17 points):
- Face: nose, left_eye, right_eye, left_ear, right_ear
- Body: left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
- Hip: left_hip, right_hip
- Legs: left_knee, right_knee, left_ankle, right_ankle

#### 1.3 MediaPipe Tasks Vision

```typescript
import { FaceLandmarker, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';

// Advanced landmarking for fine-grained tracking
const faceLandmarker = await FaceLandmarker.createFromOptions(/* ... */);
const handLandmarker = await HandLandmarker.createFromOptions(/* ... */);
```

#### 1.4 React Integration

```typescript
// Custom hook for real-time camera feed
function useCameraStream(): MediaStream | null {
  const [stream, setStream] = useState<MediaStream | null>(null);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(setStream);
    return () => stream?.getTracks().forEach(t => t.stop());
  }, []);

  return stream;
}

// Canvas overlay component for bounding boxes
function DetectionOverlay({ detections, videoRef }: Props) {
  // Draw bounding boxes on canvas overlay
}
```

### Data Flow

```
Browser Camera → getUserMedia → TensorFlow.js → Detection Events
                                        ↓
                                  WebSocket
                                        ↓
                               React Dashboard (real-time)
```

---

## 2. Uploaded Video Processing

### Technology Stack

| Component | Library/Service | Purpose |
|-----------|-----------------|---------|
| API Layer | FastAPI | Video upload endpoint, job management |
| Detection | Ultralytics YOLOv8 | Server-side object detection |
| Storage | S3 | Video file storage |
| Streaming | Kinesis Data Streams | Event emission |
| Queue | SQS | Async job processing |

### Implementation Details

#### 2.1 FastAPI Service

```python
from fastapi import FastAPI, UploadFile, File
import boto3
from ultralytics import YOLO

app = FastAPI()
s3 = boto3.client("s3")
yolo = YOLO("yolov8n.pt")  # Nano model for speed

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    # Upload to S3
    s3.upload_fileobj(file.file, "videos-bucket", file.filename)

    # Trigger async processing
    # Extract frames at 1fps, run YOLOv8, emit events

    return {"job_id": "uuid", "status": "processing"}
```

#### 2.2 Frame Extraction Pipeline

```python
import cv2

def extract_frames_at_1fps(video_path: str, output_dir: str):
    """Extract frames at 1 frame per second."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(fps)))

    frame_idx = 0
    extracted = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            cv2.imwrite(f"{output_dir}/frame_{extracted:06d}.jpg", frame)
            extracted += 1

        frame_idx += 1

    cap.release()
    return extracted
```

#### 2.3 YOLOv8 Detection

```python
from ultralytics import YOLO

def detect_objects(frame_path: str) -> list[dict]:
    results = yolo(frame_path, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detections.append({
                "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                "confidence": float(box.conf[0]),
                "class": result.names[int(box.cls[0])]
            })

    return detections
```

### Data Flow

```
Video Upload → S3 → Lambda Trigger → Frame Extraction
                                           ↓
                                     YOLOv8 Detection
                                           ↓
                              Kinesis Data Streams → React Dashboard
```

---

## 3. Edge Deployment (AWS IoT Greengrass)

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Edge Runtime | AWS IoT Greengrass v2 | Local inference orchestration |
| Model Format | YOLOv8 ONNX | Cross-platform, Greengrass-compatible |
| Inference | Greengrass ML Component | Local object detection |
| Sync | IoT Core | Event backhaul to cloud |

### Implementation Details

#### 3.1 YOLOv8 ONNX Export

```bash
# Export YOLOv8 to ONNX format for Greengrass
yolo export --model yolov8n.pt --format onnx --opset 12

# ONNX model can be deployed as Greengrass ML component
```

#### 3.2 Greengrass Component Structure

```
com.openvision.edge.detector/
├── artifact/
│   ├── yolov8n.onnx          # Model file
│   └── inference.py          # Lambda function
├── component.yaml            # Recipe
└── tests/
```

#### 3.3 Edge Inference Lambda

```python
import onnxruntime as ort
import numpy as np

class EdgeDetector:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path)

    def detect(self, frame: np.ndarray) -> list[dict]:
        # Preprocess, inference, postprocess
        # Emit events locally and to cloud via IoT Core
```

### Architecture

```
┌─────────────────────────────────┐
│     Edge Device (Camera)        │
│  ┌───────────────────────────┐  │
│  │  Greengrass ML Component   │  │
│  │  ┌─────────────────────┐  │  │
│  │  │   YOLOv8 ONNX       │  │  │
│  │  └─────────────────────┘  │  │
│  │  ┌─────────────────────┐  │  │
│  │  │  Local Inference    │  │  │
│  │  └─────────────────────┘  │  │
│  └───────────────────────────┘  │
│              │                   │
│              │ IoT Core          │
└──────────────┼───────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│         AWS Cloud               │
│  ┌───────────────────────────┐  │
│  │     Kinesis Data Streams  │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │     React Dashboard       │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

---

## 4. React Dashboard Integration

### Technology Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| State | React Hooks + Context | Detection state management |
| Real-time | WebSocket (native) | Live event streaming |
| Visualization | Canvas API | Bounding box overlay |
| Styling | Tailwind CSS | UI components |

### Implementation Details

#### 4.1 Real-time Detection Hook

```typescript
interface DetectionEvent {
  camera_id: string;
  timestamp: string;
  labels: Array<{
    name: string;
    confidence: number;
    bounding_box: {
      width: number;
      height: number;
      left: number;
      top: number;
    } | null;
  }>;
  frame_time: number;
}

function useRealtimeDetections(wsUrl: string) {
  const [detections, setDetections] = useState<DetectionEvent[]>([]);

  useEffect(() => {
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data: DetectionEvent = JSON.parse(event.data);
      setDetections(prev => [...prev.slice(-100), data]); // Keep last 100
    };

    return () => ws.close();
  }, [wsUrl]);

  return detections;
}
```

#### 4.2 Canvas Overlay Component

```typescript
interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  detections: DetectionEvent[];
}

function DetectionOverlay({ videoRef, detections }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !videoRef.current) return;

    // Match canvas size to video
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    // Clear and draw
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const detection of detections) {
      for (const label of detection.labels) {
        if (!label.bounding_box) continue;

        const { left, top, width, height } = label.bounding_box;
        const x = left * canvas.width;
        const y = top * canvas.height;
        const w = width * canvas.width;
        const h = height * canvas.height;

        // Draw rectangle
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        // Draw label
        ctx.fillStyle = "#00FF00";
        ctx.fillText(`${label.name} ${Math.round(label.confidence)}%`, x, y - 5);
      }
    }
  }, [detections, videoRef]);

  return <canvas ref={canvasRef} className="absolute inset-0" />;
}
```

---

## 5. Event Schema

### Detection Event Format

```typescript
interface DetectionEvent {
  camera_id: string;           // e.g., "cam-01", "edge-device-123"
  timestamp: string;            // ISO 8601 format
  labels: Array<{
    name: string;               // e.g., "Person", "Phone", "Laptop"
    confidence: number;         // 0-100 percentage
    bounding_box: {
      width: number;            // Relative 0-1
      height: number;           // Relative 0-1
      left: number;             // Relative 0-1
      top: number;              // Relative 0-1
    } | null;                   // null for frame-level detections
  }>;
  frame_time: number;          // Seconds from video start
}
```

### WebSocket Message Types

```typescript
type WSMessage =
  | { type: "detection"; data: DetectionEvent }
  | { type: "alert"; data: AlertEvent }
  | { type: "status"; data: StatusEvent };

interface AlertEvent {
  camera_id: string;
  alert_type: "phone_detected" | "multiple_persons" | "no_detection";
  severity: "low" | "medium" | "high";
  message: string;
}
```

---

## 6. Infrastructure

### AWS Resources

| Resource | Service | Purpose |
|----------|---------|---------|
| Video Storage | S3 | Uploaded videos, frame outputs |
| Stream Processing | Kinesis Data Streams | Real-time event streaming |
| Job Queue | SQS | Async video processing |
| Serverless | Lambda (existing) | Rekognition processor |
| Server-side CV | Lambda + FastAPI | YOLOv8 processing |
| Edge Runtime | IoT Greengrass v2 | Edge inference |
| Model Storage | S3 + ECR | YOLOv8 ONNX, Docker images |
| ML Inference | SageMaker (optional) | Cloud fallback |

### S3 Bucket Structure

```
s3://openvision-videos/
├── videos/              # Original uploads (trigger source)
│   └── {camera_id}/
│       └── {timestamp}.mp4
├── frames/              # Extracted frames
│   └── {job_id}/
│       └── frame_000001.jpg
└── processed/           # Final outputs
    └── {job_id}/
        └── detections.json
```

---

## 7. Development Status

| Feature | Status | Owner |
|---------|--------|-------|
| Rekognition Lambda (existing) | Implemented | Team |
| Demo Processor Lambda (existing) | Implemented | Team |
| TensorFlow.js Integration | Planned | TBD |
| MediaPipe Integration | Planned | TBD |
| FastAPI + YOLOv8 Service | Planned | TBD |
| WebSocket Real-time Service | Planned | TBD |
| Canvas Overlay Component | Planned | TBD |
| Edge (Greengrass) Deployment | Planned | TBD |

---

## 8. Team Task Assignment

### Frontend Team
- [ ] TensorFlow.js integration hooks (`useObjectDetection`, `usePoseDetection`)
- [ ] MediaPipe Tasks Vision integration
- [ ] WebSocket client for real-time events
- [ ] Canvas overlay component for bounding boxes
- [ ] React dashboard state management

### Backend Team
- [ ] FastAPI service for video upload
- [ ] YOLOv8 integration with frame extraction
- [ ] Kinesis event emission
- [ ] SQS job queue setup

### DevOps Team
- [ ] AWS IoT Greengrass component creation
- [ ] YOLOv8 ONNX export pipeline
- [ ] Edge device provisioning
- [ ] CI/CD for model updates

### ML Team
- [ ] Model selection (YOLOv8n vs YOLOv8s vs YOLOv8m)
- [ ] Custom class training for domain-specific objects
- [ ] Model performance benchmarking
- [ ] Edge inference optimization