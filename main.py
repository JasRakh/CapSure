import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2


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
    
    args = parser.parse_args()
    
    if args.train:
        print("=" * 60)
        print("TRAINING MODE")
        print("=" * 60)
        from src.train import main as train_main
        import sys
        sys.argv = ['train.py', '--epochs', str(args.epochs)]
        train_main()
        return
    
    print("=" * 60)
    print("PILL AND CAPSULE DETECTION SYSTEM")
    print("=" * 60)
    
    if os.path.exists(args.weights):
        print(f"✓ Loading custom model from {args.weights}")
        model = YOLO(args.weights)
    else:
        print(f"⚠ Custom model not found at {args.weights}")
        print("  Using pretrained YOLOv8n model (trained on COCO dataset)")
        print("  Note: This model doesn't recognize pills. Train a custom model for accurate detection.")
        print("  Run: python main.py --train")
        model = YOLO('yolov8n.pt')
    
    if args.webcam:
        source = 0
        print(f"\n📹 Starting webcam detection...")
        print("   Press 'q' to quit")
    elif args.image:
        if not os.path.exists(args.image):
            print(f"❌ Error: Image not found at {args.image}")
            return
        source = args.image
        print(f"\n🖼️  Processing image: {args.image}")
    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"❌ Error: Directory not found at {args.dir}")
            return
        source = args.dir
        print(f"\n📁 Processing directory: {args.dir}")
    else:
        images_dir = Path('data/images')
        if images_dir.exists():
            images = list(images_dir.rglob('*.jpg')) + list(images_dir.rglob('*.png')) + list(images_dir.rglob('*.jpeg'))
            if images:
                source = [str(img) for img in images]
                print(f"\n📁 Processing {len(images)} images from data/images/")
            else:
                print("❌ No images found in data/images/")
                print("   Add images or use --image or --dir options")
                return
        else:
            print("❌ No source specified and data/images/ doesn't exist")
            print("   Use --image, --dir, or --webcam")
            return
    
    if args.save:
        os.makedirs('results', exist_ok=True)
        print(f"💾 Results will be saved to results/")
    
    print(f"\n🔍 Running detection (confidence threshold: {args.conf})...")
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
        print("DETECTION SUMMARY")
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
                print(f"\n📸 {img_name}:")
                for j, box in enumerate(result.boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    print(f"   {j+1}. {class_name} (confidence: {conf:.2f})")
            else:
                print(f"\n📸 {img_name}: No detections")
        
        print(f"\n✓ Total detections: {total_detections}")
        print(f"✓ Detection complete!")
        
        if args.save:
            print(f"\n💾 Results saved to: results/detections/")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Detection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during detection: {e}")
        raise


if __name__ == '__main__':
    main()
