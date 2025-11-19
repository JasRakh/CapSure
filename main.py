import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import kagglehub


def main():
    parser = argparse.ArgumentParser(
        description='Pill and Capsule Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --image data/images/pill_sample_1.jpg
  python main.py --dir data/images/
  python main.py --webcam
  python main.py --train --epochs 100
  python main.py --download-dataset
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--dir', type=str, help='Path to directory with images')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for detection')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--weights', type=str, default='models/best.pt', help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (for training)')
    parser.add_argument('--save', action='store_true', default=True, help='Save detection results')
    parser.add_argument('--show', action='store_true', help='Show detection results in window')
    parser.add_argument('--download-dataset', action='store_true', help='Download dataset from Kaggle')
    parser.add_argument('--dataset', type=str, default='tommyngx/epillid-data-v1', help='Kaggle dataset identifier')
    
    args = parser.parse_args()
    
    if args.download_dataset:
        print("=" * 60)
        print("Downloading Dataset from Kaggle")
        print("=" * 60)
        try:
            print(f"Downloading dataset: {args.dataset}")
            path = kagglehub.dataset_download(args.dataset)
            print(f"Path to dataset files: {path}")
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Make sure you have Kaggle credentials configured.")
            print("See: https://github.com/Kaggle/kaggle-api#api-credentials")
        return
    
    if args.train:
        print("=" * 60)
        print("Training Mode")
        print("=" * 60)
        from src.train import main as train_main
        import sys
        sys.argv = ['train.py', '--epochs', str(args.epochs)]
        train_main()
        return
    
    print("=" * 60)
    print("Pill and Capsule Detection System")
    print("=" * 60)
    
    if os.path.exists(args.weights):
        print(f"Loading custom model from {args.weights}")
        model = YOLO(args.weights)
    else:
        print(f"Custom model not found at {args.weights}")
        print("  Using pretrained YOLOv8n model (trained on COCO dataset)")
        print("  Note: This model doesn't recognize pills. Train a custom model for accurate detection.")
        print("  Run: python main.py --train")
        model = YOLO('yolov8n.pt')
    
    if args.webcam:
        source = 0
        print(f"\nStarting webcam detection...")
        print("   Press 'q' to quit")
    elif args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return
        source = args.image
        print(f"\nProcessing image: {args.image}")
    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"Error: Directory not found at {args.dir}")
            return
        source = args.dir
        print(f"\nProcessing directory: {args.dir}")
    else:
        images_dir = Path('data/images')
        if images_dir.exists():
            images = list(images_dir.rglob('*.jpg')) + list(images_dir.rglob('*.png')) + list(images_dir.rglob('*.jpeg'))
            if images:
                source = [str(img) for img in images]
                print(f"\nProcessing {len(images)} images from data/images/")
            else:
                print("No images found in data/images/")
                print("   Add images or use --image or --dir options")
                return
        else:
            print("No source specified and data/images/ doesn't exist")
            print("   Use --image, --dir, or --webcam")
            return
    
    if args.save:
        os.makedirs('results', exist_ok=True)
        print(f"Results will be saved to results/")
    
    print(f"\nRunning detection (confidence threshold: {args.conf})...")
    print("-" * 60)
    
    try:
        results = model.predict(
            source=source,
            conf=args.conf,
            save=args.save,
            show=args.show,
            project='results' if args.save else None,
            name='detections'
        )
        
        print("\n" + "=" * 60)
        print("Detection Summary")
        print("=" * 60)
        
        total_detections = 0
        for i, result in enumerate(results):
            if hasattr(result, 'path'):
                img_name = Path(result.path).name
            else:
                img_name = f"Image {i+1}"
            
            if result.boxes is not None and len(result.boxes) > 0:
                detections = len(result.boxes)
                total_detections += detections
                print(f"\n{img_name}:")
                for j, box in enumerate(result.boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    print(f"   {j+1}. {class_name} (confidence: {conf:.2f})")
            else:
                print(f"\n{img_name}: No detections")
        
        print(f"\nTotal detections: {total_detections}")
        print("Done")
        
        if args.save:
            print(f"\nResults saved to: results/detections/")
        
    except KeyboardInterrupt:
        print("\n\nDetection interrupted by user")
    except Exception as e:
        print(f"\nError during detection: {e}")
        raise


if __name__ == '__main__':
    main()
