import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2


def main():
    parser = argparse.ArgumentParser(description='Detect pills and capsules in images/videos')
    parser.add_argument('--source', type=str, required=True, 
                       help='Path to image, video, or webcam (0 for webcam)')
    parser.add_argument('--weights', type=str, default='models/best.pt',
                       help='Path to model weights file')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for inference')
    parser.add_argument('--save', action='store_true',
                       help='Save detection results')
    parser.add_argument('--show', action='store_true', default=True,
                       help='Show detection results')
    
    args = parser.parse_args()
    
    if args.save:
        os.makedirs('results', exist_ok=True)
    
    if not os.path.exists(args.weights):
        print(f"Model weights not found at {args.weights}")
        print("Using pretrained YOLOv8n model for demonstration...")
        model = YOLO('yolov8n.pt')
    else:
        print(f"Loading model from {args.weights}")
        model = YOLO(args.weights)
    
    source = args.source
    if source == '0' or source == 0:
        source = 0
    
    print(f"Running detection on: {source}")
    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
        show=args.show
    )
    
    if isinstance(source, str) and os.path.exists(source):
        for result in results:
            print(f"\nDetections in {result.path}:")
            if result.boxes is not None:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    print(f"  - {class_name}: {conf:.2f}")
            else:
                print("  No detections found")
    
    print("\nDetection complete!")


if __name__ == '__main__':
    main()
