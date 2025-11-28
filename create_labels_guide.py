#!/usr/bin/env python3
"""
Guide and utilities for creating YOLO format labels.
This script helps you understand the label format and provides tools to work with labels.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def show_label_format_example():
    """Show example of YOLO label format."""
    print("=" * 60)
    print("YOLO Label Format")
    print("=" * 60)
    print("""
Each image needs a corresponding .txt file with the same name.

Format: class_id x_center y_center width height

Where:
  - class_id: 0 for 'pill', 1 for 'capsule'
  - x_center: X coordinate of bounding box center (normalized 0-1)
  - y_center: Y coordinate of bounding box center (normalized 0-1)
  - width: Width of bounding box (normalized 0-1)
  - height: Height of bounding box (normalized 0-1)

Example label file (image.jpg -> image.txt):
  0 0.5 0.5 0.3 0.4
  1 0.7 0.3 0.2 0.25

This means:
  - Pill (class 0) at center (0.5, 0.5) with size 0.3x0.4
  - Capsule (class 1) at center (0.7, 0.3) with size 0.2x0.25
""")


def convert_bbox_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height):
    """
    Convert bounding box from pixel coordinates to YOLO format.
    
    Args:
        x_min, y_min: Top-left corner (pixels)
        x_max, y_max: Bottom-right corner (pixels)
        img_width, img_height: Image dimensions (pixels)
    
    Returns:
        tuple: (x_center, y_center, width, height) normalized
    """
    x_center = ((x_min + x_max) / 2.0) / img_width
    y_center = ((y_min + y_max) / 2.0) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return x_center, y_center, width, height


def convert_yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """
    Convert YOLO format to pixel coordinates.
    
    Args:
        x_center, y_center, width, height: Normalized YOLO coordinates
        img_width, img_height: Image dimensions (pixels)
    
    Returns:
        tuple: (x_min, y_min, x_max, y_max) in pixels
    """
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    x_min = int(x_center_px - width_px / 2)
    y_min = int(y_center_px - height_px / 2)
    x_max = int(x_center_px + width_px / 2)
    y_max = int(y_center_px + height_px / 2)
    
    return x_min, y_min, x_max, y_max


def visualize_labels(image_path, label_path=None):
    """
    Visualize labels on an image.
    
    Args:
        image_path: Path to image file
        label_path: Path to label file (optional, will try to find automatically)
    """
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    # Try to find label file
    if label_path is None:
        label_path = img_path.parent.parent / 'labels' / img_path.parent.name / (img_path.stem + '.txt')
    else:
        label_path = Path(label_path)
    
    if not label_path.exists():
        print(f"No label file found at: {label_path}")
        print("Creating empty label file template...")
        return
    
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not read image: {image_path}")
        return
    
    img_height, img_width = img.shape[:2]
    
    # Read labels
    class_names = ['pill', 'capsule']
    colors = [(0, 255, 0), (255, 0, 0)]  # Green for pill, Red for capsule
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to pixel coordinates
                x_min, y_min, x_max, y_max = convert_yolo_to_bbox(
                    x_center, y_center, width, height, img_width, img_height
                )
                
                # Draw bounding box
                color = colors[class_id] if class_id < len(colors) else (255, 255, 255)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Draw label
                label_text = f"{class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'}"
                cv2.putText(img, label_text, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Show image
    print(f"Displaying: {img_path.name}")
    print("Press any key to close the window")
    cv2.imshow('Labeled Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def validate_label_file(label_path, image_path=None):
    """
    Validate a YOLO label file.
    
    Args:
        label_path: Path to label file
        image_path: Optional path to corresponding image
    """
    label_file = Path(label_path)
    if not label_file.exists():
        print(f"Error: Label file not found: {label_path}")
        return False
    
    errors = []
    warnings = []
    
    with open(label_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"Line {line_num}: Expected 5 values, got {len(parts)}")
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Validate class ID
                if class_id not in [0, 1]:
                    errors.append(f"Line {line_num}: Invalid class_id {class_id} (must be 0 or 1)")
                
                # Validate coordinates
                if not (0 <= x_center <= 1):
                    errors.append(f"Line {line_num}: x_center {x_center} out of range [0, 1]")
                if not (0 <= y_center <= 1):
                    errors.append(f"Line {line_num}: y_center {y_center} out of range [0, 1]")
                if not (0 < width <= 1):
                    errors.append(f"Line {line_num}: width {width} out of range (0, 1]")
                if not (0 < height <= 1):
                    errors.append(f"Line {line_num}: height {height} out of range (0, 1]")
                
                # Check if box is reasonable
                if width < 0.01 or height < 0.01:
                    warnings.append(f"Line {line_num}: Very small bounding box (width={width}, height={height})")
                if width > 0.95 or height > 0.95:
                    warnings.append(f"Line {line_num}: Very large bounding box (width={width}, height={height})")
                
            except ValueError as e:
                errors.append(f"Line {line_num}: Invalid number format - {e}")
    
    if errors:
        print("Errors found:")
        for error in errors:
            print(f"  ✗ {error}")
        return False
    
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    
    print("✓ Label file is valid!")
    return True


def main():
    parser = argparse.ArgumentParser(description='YOLO Label Format Guide and Utilities')
    parser.add_argument('--format', action='store_true', help='Show label format example')
    parser.add_argument('--validate', type=str, help='Validate a label file')
    parser.add_argument('--visualize', type=str, help='Visualize labels on an image')
    parser.add_argument('--label-file', type=str, help='Label file path (for --visualize)')
    
    args = parser.parse_args()
    
    if args.format:
        show_label_format_example()
    elif args.validate:
        validate_label_file(args.validate)
    elif args.visualize:
        visualize_labels(args.visualize, args.label_file)
    else:
        print("=" * 60)
        print("YOLO Label Creation Guide")
        print("=" * 60)
        print("\nYou need to create labels for your images to train the model.")
        print("\nOptions:")
        print("\n1. Use LabelImg (Recommended GUI tool):")
        print("   Install: pip install labelImg")
        print("   Run: labelImg")
        print("   - Set format to YOLO")
        print("   - Classes: pill (0), capsule (1)")
        print("   - Open data/images/train/ and annotate images")
        print("\n2. Use online tools:")
        print("   - Roboflow: https://roboflow.com/annotate")
        print("   - CVAT: https://cvat.org/")
        print("   - Label Studio: https://labelstud.io/")
        print("\n3. Create labels manually:")
        print("   - Each image needs a .txt file with same name")
        print("   - Format: class_id x_center y_center width height")
        print("   - All values normalized 0-1")
        print("\nCommands:")
        print("  python create_labels_guide.py --format          # Show format example")
        print("  python create_labels_guide.py --validate <file>  # Validate label file")
        print("  python create_labels_guide.py --visualize <img>  # Visualize labels")
        print("=" * 60)


if __name__ == '__main__':
    main()

