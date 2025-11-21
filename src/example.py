from ultralytics import YOLO
import cv2
from pathlib import Path
import os


def example_detection():
    model_path = 'models/best.pt'
    if os.path.exists(model_path):
        print(f"Loading custom model from {model_path}")
        model = YOLO(model_path)
    else:
        print("Custom model not found. Using pretrained YOLOv11n for demonstration.")
        print("Train your own model using: python src/train.py")
        model = YOLO('yolov11n.pt')
    
    image_path = 'data/raw/example.jpg'
    if os.path.exists(image_path):
        print(f"\nRunning detection on {image_path}")
        results = model.predict(
            source=image_path,
            conf=0.25,
            save=True,
            show=False
        )
        
        for result in results:
            print(f"\nDetections in {result.path}:")
            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    print(f"  Detection {i+1}: {class_name} (confidence: {conf:.2f})")
                    print(f"    Bounding box: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")
            else:
                print("  No detections found")
    else:
        print(f"\nExample image not found at {image_path}")
        print("Please add images to data/raw/ directory")


def example_webcam():
    model_path = 'models/best.pt'
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        model = YOLO('yolov11n.pt')
    
    print("\nStarting webcam detection...")
    print("Press 'q' to quit")
    
    results = model.predict(
        source=0,
        conf=0.25,
        show=True,
        stream=True
    )
    
    for result in results:
        pass


def example_batch_detection():
    model_path = 'models/best.pt'
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        model = YOLO('yolov11n.pt')
    
    raw_dir = Path('data/raw')
    if raw_dir.exists():
        image_files = list(raw_dir.glob('*.jpg')) + list(raw_dir.glob('*.png'))
        
        if image_files:
            print(f"\nProcessing {len(image_files)} images...")
            
            results = model.predict(
                source=[str(f) for f in image_files],
                conf=0.25,
                save=True
            )
            
            total_detections = 0
            for result in results:
                if result.boxes is not None:
                    total_detections += len(result.boxes)
            
            print(f"\nTotal detections: {total_detections}")
        else:
            print("No images found in data/raw/")
    else:
        print("data/raw/ directory not found")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Example detection scripts')
    parser.add_argument('--mode', type=str, choices=['image', 'webcam', 'batch'],
                       default='image', help='Detection mode')
    
    args = parser.parse_args()
    
    if args.mode == 'image':
        example_detection()
    elif args.mode == 'webcam':
        example_webcam()
    elif args.mode == 'batch':
        example_batch_detection()
