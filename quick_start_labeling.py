#!/usr/bin/env python3
"""
Quick start script to help you begin labeling.
This script will help you create labels for a small subset to test training.
"""

from pathlib import Path
import shutil


def create_sample_labels_for_testing(num_images=10):
    """
    Create empty label files for a small number of images to test the training pipeline.
    You'll need to fill these with actual annotations later.
    """
    train_images_dir = Path('data/images/train')
    train_labels_dir = Path('data/labels/train')
    val_images_dir = Path('data/images/val')
    val_labels_dir = Path('data/labels/val')
    
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    train_images = sorted(list(train_images_dir.glob('*.jpg')) + 
                         list(train_images_dir.glob('*.png')))
    val_images = sorted(list(val_images_dir.glob('*.jpg')) + 
                       list(val_images_dir.glob('*.png')))
    
    print("=" * 60)
    print("Quick Start Labeling Helper")
    print("=" * 60)
    print(f"\nFound {len(train_images)} training images")
    print(f"Found {len(val_images)} validation images")
    
    if len(train_images) == 0:
        print("\n❌ No training images found!")
        print("   Make sure you have images in data/images/train/")
        return False
    
    # Create empty label files for first N images
    created = 0
    print(f"\nCreating empty label files for first {num_images} training images...")
    print("   (You'll need to add actual annotations to these files)")
    
    for img_path in train_images[:num_images]:
        label_path = train_labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            # Create empty file
            label_path.write_text("")
            created += 1
    
    # Create a few for validation too
    val_created = 0
    for img_path in val_images[:min(5, len(val_images))]:
        label_path = val_labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            label_path.write_text("")
            val_created += 1
    
    print(f"✓ Created {created} empty label files in data/labels/train/")
    print(f"✓ Created {val_created} empty label files in data/labels/val/")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("\n1. Install LabelImg (recommended):")
    print("   pip install labelImg")
    print("   labelImg")
    print("\n2. Or manually edit label files:")
    print("   Format: class_id x_center y_center width height")
    print("   Example: 0 0.5 0.5 0.3 0.4  (pill at center)")
    print("\n3. Label at least a few images to test training")
    print("\n4. Then run: python main.py --train --epochs 10")
    print("=" * 60)
    
    return True


def show_labeling_tools():
    """Show information about labeling tools."""
    print("=" * 60)
    print("Labeling Tools Options")
    print("=" * 60)
    print("\n1. LabelImg (GUI - Easiest):")
    print("   Install: pip install labelImg")
    print("   Run: labelImg")
    print("   - Set format to YOLO")
    print("   - Open: data/images/train/")
    print("   - Save to: data/labels/train/")
    print("\n2. Online Tools:")
    print("   - Roboflow: https://roboflow.com/annotate")
    print("   - CVAT: https://cvat.org/")
    print("\n3. Manual (for small datasets):")
    print("   Edit .txt files directly")
    print("   Format: class_id x_center y_center width height")
    print("   All values normalized 0-1")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Quick start labeling helper')
    parser.add_argument('--create-empty', type=int, default=10,
                       help='Create empty label files for N images (default: 10)')
    parser.add_argument('--show-tools', action='store_true',
                       help='Show information about labeling tools')
    
    args = parser.parse_args()
    
    if args.show_tools:
        show_labeling_tools()
    else:
        create_sample_labels_for_testing(args.create_empty)


if __name__ == '__main__':
    main()

