import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil
from sklearn.model_selection import train_test_split


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


def prepare_dataset_from_test_images(data_dir='data', train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split images from test directory into train/val/test splits.
    Also copies corresponding label files if they exist.
    
    Args:
        data_dir: Path to data directory
        train_ratio: Ratio of images for training
        val_ratio: Ratio of images for validation
        test_ratio: Ratio of images for testing
    """
    data_path = Path(data_dir)
    test_images_dir = data_path / 'images' / 'test'
    test_labels_dir = data_path / 'labels' / 'test'
    
    train_images_dir = data_path / 'images' / 'train'
    train_labels_dir = data_path / 'labels' / 'train'
    val_images_dir = data_path / 'images' / 'val'
    val_labels_dir = data_path / 'labels' / 'val'
    
    # Create directories
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    if not test_images_dir.exists():
        print(f"Error: Test images directory not found at {test_images_dir}")
        return False
    
    # Find all images in test directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(test_images_dir.glob(f'*{ext}')))
        test_images.extend(list(test_images_dir.glob(f'*{ext.upper()}')))
    
    if not test_images:
        print(f"No images found in {test_images_dir}")
        return False
    
    print(f"Found {len(test_images)} images in test directory")
    
    # Split images
    train_imgs, temp_imgs = train_test_split(test_images, test_size=(1 - train_ratio), random_state=42)
    val_size = val_ratio / (val_ratio + test_ratio)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(1 - val_size), random_state=42)
    
    print(f"\nSplitting images:")
    print(f"  Train: {len(train_imgs)} images")
    print(f"  Val: {len(val_imgs)} images")
    print(f"  Test: {len(test_imgs)} images")
    
    # Copy images and labels
    for split_name, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
        images_dir = data_path / 'images' / split_name
        labels_dir = data_path / 'labels' / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        labels_copied = 0
        for img_path in imgs:
            # Copy image (skip if source and destination are the same)
            dest_img = images_dir / img_path.name
            if img_path.resolve() != dest_img.resolve():
                shutil.copy2(img_path, dest_img)
                copied += 1
            else:
                # File is already in the correct location
                copied += 1
            
            # Copy label if exists
            label_name = img_path.stem + '.txt'
            source_label = test_labels_dir / label_name
            if source_label.exists():
                dest_label = labels_dir / label_name
                if source_label.resolve() != dest_label.resolve():
                    shutil.copy2(source_label, dest_label)
                    labels_copied += 1
                else:
                    # Label is already in the correct location
                    labels_copied += 1
        
        print(f"  {split_name}: {copied} images, {labels_copied} labels")
    
    print("\nDataset prepared successfully!")
    return True


def validate_dataset(data_dir='data'):
    """
    Validate that dataset has required structure for training.
    
    Returns:
        tuple: (is_valid, message)
    """
    data_path = Path(data_dir)
    
    # Check directories
    train_img_dir = data_path / 'images' / 'train'
    val_img_dir = data_path / 'images' / 'val'
    train_label_dir = data_path / 'labels' / 'train'
    val_label_dir = data_path / 'labels' / 'val'
    
    if not train_img_dir.exists():
        return False, f"Train images directory not found: {train_img_dir}"
    
    if not val_img_dir.exists():
        return False, f"Val images directory not found: {val_img_dir}"
    
    # Count images
    train_images = list(train_img_dir.glob('*.jpg')) + list(train_img_dir.glob('*.png')) + \
                   list(train_img_dir.glob('*.jpeg'))
    val_images = list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png')) + \
                 list(val_img_dir.glob('*.jpeg'))
    
    if len(train_images) == 0:
        return False, f"No training images found in {train_img_dir}"
    
    if len(val_images) == 0:
        return False, f"No validation images found in {val_img_dir}"
    
    # Count labels
    train_labels = list(train_label_dir.glob('*.txt')) if train_label_dir.exists() else []
    val_labels = list(val_label_dir.glob('*.txt')) if val_label_dir.exists() else []
    
    if len(train_labels) == 0:
        return False, f"No training labels found in {train_label_dir}. Labels are required for training."
    
    if len(val_labels) == 0:
        return False, f"No validation labels found in {val_label_dir}. Labels are required for training."
    
    return True, f"Dataset valid: {len(train_images)} train images, {len(val_images)} val images, {len(train_labels)} train labels, {len(val_labels)} val labels"


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for pill/capsule detection')
    parser.add_argument('--model', type=str, default='yolov8n',
                       help='Base model (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, or yolov11n if available)')
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
    
    parser.add_argument('--prepare-dataset', action='store_true',
                       help='Split test images into train/val/test before training')
    
    args = parser.parse_args()
    
    # Prepare dataset if requested
    if args.prepare_dataset:
        print("=" * 60)
        print("Preparing Dataset from Test Images")
        print("=" * 60)
        if not prepare_dataset_from_test_images():
            print("Failed to prepare dataset. Exiting.")
            return
    
    # Validate dataset
    print("=" * 60)
    print("Validating Dataset")
    print("=" * 60)
    is_valid, message = validate_dataset()
    if not is_valid:
        print(f"Dataset validation failed: {message}")
        print("\nOptions:")
        print("  1. Add images and labels to data/images/train and data/images/val")
        print("  2. Run with --prepare-dataset to split test images into train/val/test")
        print("     Note: This requires labels in data/labels/test/")
        return
    print(message)
    
    if not os.path.exists(args.data):
        print(f"data.yaml not found at {args.data}")
        print("Creating default data.yaml...")
        create_data_yaml()
        args.data = 'data/data.yaml'
    
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    
    # Handle model loading - remove .pt extension for auto-download
    model_name = args.model
    if model_name.endswith('.pt'):
        model_name = model_name.replace('.pt', '')
    
    # Check if it's a local file
    if Path(args.model).exists():
        model_name = args.model
        print(f"Loading local model: {model_name}")
    else:
        print(f"Loading model: {model_name} (will auto-download if needed)")
    
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative model...")
        # Fallback to YOLOv8 if YOLOv11 not available
        if 'yolov11' in model_name.lower():
            print("YOLOv11 may not be available. Trying YOLOv8...")
            model_name = model_name.replace('yolov11', 'yolov8')
            model = YOLO(model_name)
        else:
            raise
    
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
    
    results_dir = Path(args.project) / args.name
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Results saved to: {results_dir.absolute()}")
    print("\nTo view training results, run:")
    print(f"  python view_training_results.py")
    print(f"  python view_training_results.py --open  # Opens in file manager")
    print("\nKey files in results directory:")
    print(f"  - results.csv: Training metrics per epoch")
    print(f"  - results.png: Training loss and metric curves")
    print(f"  - confusion_matrix.png: Confusion matrix")
    print(f"  - F1_curve.png: F1 score curve")
    print(f"  - PR_curve.png: Precision-Recall curve")
    print(f"  - weights/best.pt: Best model weights")
    print(f"  - weights/last.pt: Last epoch weights")
    print("=" * 60)


if __name__ == '__main__':
    main()
