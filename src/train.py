import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import yaml


def create_data_yaml(data_dir='data', classes=None):
    if classes is None:
        classes = ['pill', 'capsule']
    
    data_yaml = {
        'path': str(Path(data_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = Path(data_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created data.yaml at {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 model for pill/capsule detection')
    parser.add_argument('--model', type=str, default='yolov11n.pt',
                       help='Base model (yolov11n.pt, yolov11s.pt, yolov11m.pt, yolov11l.pt, yolov11x.pt)')
    parser.add_argument('--data', type=str, default='data/data.yaml',
                       help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (cuda, cpu, or leave empty for auto)')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='pill_capsule_detection',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"data.yaml not found at {args.data}")
        print("Creating default data.yaml...")
        create_data_yaml()
        args.data = 'data/data.yaml'
    
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Data: {args.data}\n")
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        plots=True
    )
    
    os.makedirs('models', exist_ok=True)
    best_model_path = Path(args.project) / args.name / 'weights' / 'best.pt'
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, 'models/best.pt')
        print(f"\nBest model saved to models/best.pt")
    
    print("\nTraining complete!")
    print(f"Results saved to: {Path(args.project) / args.name}")


if __name__ == '__main__':
    main()
