from pathlib import Path
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import json
import re


# Class mapping: pill=0, capsule=1
CLASS_MAPPING = {
    'pill': 0,
    'capsule': 1,
    'pills': 0,
    'capsules': 1,
    'tablet': 0,
    'tablets': 0,
    '0': 0,
    '1': 1,
    0: 0,
    1: 1
}


def download_and_prepare_dataset(dataset_id="tommyngx/epillid-data-v1"):
    """
    Download dataset from Kaggle and prepare it for YOLOv11 training.
    Creates proper YOLO format labels distinguishing between pill and capsule.
    
    Args:
        dataset_id: Kaggle dataset identifier
    """
    import kagglehub
    
    print(f"Loading dataset: {dataset_id}")
    
    # First, download the dataset to get the path
    print("Downloading dataset files...")
    dataset_path = kagglehub.dataset_download(dataset_id)
    print(f"Dataset downloaded to: {dataset_path}")
    
    dataset_dir = Path(dataset_path)
    
    # Look for common CSV files
    csv_files = list(dataset_dir.rglob("*.csv"))
    
    if csv_files:
        print(f"\nFound {len(csv_files)} CSV file(s)")
        for csv_file in csv_files:
            print(f"  - {csv_file.name}")
        
        # Try to load the main CSV file (usually the first one or one with 'train' in name)
        main_csv = None
        for csv_file in csv_files:
            if 'train' in csv_file.name.lower() or 'data' in csv_file.name.lower():
                main_csv = csv_file
                break
        
        if main_csv is None:
            main_csv = csv_files[0]
        
        print(f"\nLoading data from: {main_csv.name}")
        df = pd.read_csv(main_csv)
        print(f"Loaded {len(df)} records")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst 5 records:")
        print(df.head())
        
        # Prepare YOLO dataset structure
        prepare_yolo_dataset_from_dataframe(df, dataset_dir)
    else:
        # If no CSV files, try to organize images and labels directly
        print("\nNo CSV files found. Organizing images and labels directly...")
        organize_images_and_labels(dataset_dir)


def update_data_yaml():
    """Update data.yaml with correct class configuration."""
    import yaml
    
    data_yaml = {
        'path': str(Path('data').absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 2,
        'names': ['pill', 'capsule']
    }
    
    yaml_path = Path('data') / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\nUpdated data.yaml at {yaml_path}")
    print(f"  Classes: {data_yaml['names']}")


def prepare_yolo_dataset_from_dataframe(df, dataset_dir):
    """
    Prepare YOLO dataset from pandas DataFrame.
    Handles CSV data and converts annotations to YOLO format with proper class mapping.
    
    Args:
        df: DataFrame containing image paths and annotations
        dataset_dir: Path to the downloaded dataset directory
    """
    # Create YOLO directory structure
    data_dir = Path('data')
    for split in ['train', 'val', 'test']:
        (data_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (data_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Try to identify columns
    image_col = None
    label_col = None
    class_col = None
    
    # Common column name patterns
    for col in df.columns:
        col_lower = col.lower()
        if 'image' in col_lower or 'path' in col_lower or 'file' in col_lower:
            image_col = col
        if 'label' in col_lower or 'annotation' in col_lower or 'bbox' in col_lower:
            label_col = col
        if 'class' in col_lower or 'category' in col_lower or 'type' in col_lower:
            class_col = col
    
    if image_col is None:
        # Try to find image column by checking if values look like paths
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = str(df[col].iloc[0]) if len(df) > 0 else ""
                if any(ext in sample.lower() for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                    image_col = col
                    break
    
    if image_col is None:
        print("Warning: Could not identify image column. Using first column.")
        image_col = df.columns[0]
    
    print(f"\nUsing image column: {image_col}")
    if label_col:
        print(f"Using label column: {label_col}")
    if class_col:
        print(f"Using class column: {class_col}")
    
    # Split data into train/val/test (70/20/10)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42)
    
    print(f"\nSplitting dataset:")
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    
    # Process each split
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\nProcessing {split_name} set...")
        process_split(split_df, dataset_dir, data_dir, split_name, image_col, label_col)
    
    # Update data.yaml
    update_data_yaml()


def parse_annotation_string(annotation_str, img_width=None, img_height=None):
    """
    Parse annotation string from CSV. Handles various formats:
    - YOLO format: "class_id x_center y_center width height"
    - COCO format: [x_min, y_min, width, height] or similar
    - JSON format
    
    Args:
        annotation_str: String containing annotation data
        img_width: Image width for normalization (if needed)
        img_height: Image height for normalization (if needed)
        
    Returns:
        List of tuples: [(class_id, x_center, y_center, width, height), ...]
    """
    annotations = []
    
    try:
        # Try JSON format
        if annotation_str.strip().startswith('[') or annotation_str.strip().startswith('{'):
            data = json.loads(annotation_str)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # COCO-like format
                        if 'bbox' in item and 'category_id' in item:
                            bbox = item['bbox']  # [x, y, width, height]
                            class_id = item['category_id']
                            x_min, y_min, w, h = bbox
                            x_center = (x_min + w/2) / img_width if img_width else (x_min + w/2)
                            y_center = (y_min + h/2) / img_height if img_height else (y_min + h/2)
                            width = w / img_width if img_width else w
                            height = h / img_height if img_height else h
                            annotations.append((class_id, x_center, y_center, width, height))
        
        # Try YOLO format (space-separated)
        if not annotations:
            lines = annotation_str.strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append((class_id, x_center, y_center, width, height))
    
    except Exception as e:
        print(f"  Warning: Could not parse annotation: {e}")
    
    return annotations


def process_split(df, dataset_dir, data_dir, split_name, image_col, label_col):
    """
    Process a single split (train/val/test) of the dataset.
    Handles CSV data with annotations and converts to YOLO format.
    
    Args:
        df: DataFrame for this split
        dataset_dir: Original dataset directory
        data_dir: Target data directory
        split_name: Name of the split ('train', 'val', 'test')
        image_col: Column name containing image paths
        label_col: Column name containing labels/annotations
    """
    images_dir = data_dir / 'images' / split_name
    labels_dir = data_dir / 'labels' / split_name
    
    copied = 0
    objects_count = 0
    
    for idx, row in df.iterrows():
        try:
            # Get image path
            img_path_str = str(row[image_col])
            
            # Handle relative and absolute paths
            if os.path.isabs(img_path_str):
                img_path = Path(img_path_str)
            else:
                img_path = dataset_dir / img_path_str
            
            # Try to find the image file
            if not img_path.exists():
                # Try with different extensions
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    alt_path = img_path.with_suffix(ext)
                    if alt_path.exists():
                        img_path = alt_path
                        break
                
                # Try searching in common image directories
                if not img_path.exists():
                    for img_dir in dataset_dir.rglob('*'):
                        if img_dir.is_dir() and 'image' in img_dir.name.lower():
                            potential_path = img_dir / img_path.name
                            if potential_path.exists():
                                img_path = potential_path
                                break
            
            if img_path.exists() and img_path.is_file():
                # Copy image
                dest_img = images_dir / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Handle labels
                label_file = labels_dir / (img_path.stem + '.txt')
                
                if label_col and label_col in row:
                    label_data = row[label_col]
                    if pd.notna(label_data):
                        # Try to parse annotation
                        import cv2
                        img = cv2.imread(str(img_path))
                        img_height, img_width = img.shape[:2] if img is not None else (None, None)
                        
                        annotations = parse_annotation_string(str(label_data), img_width, img_height)
                        
                        if annotations:
                            # Convert class IDs
                            converted_annotations = []
                            for ann in annotations:
                                old_class_id, x_center, y_center, width, height = ann
                                new_class_id = convert_class_id(old_class_id)
                                converted_annotations.append((new_class_id, x_center, y_center, width, height))
                            
                            write_yolo_label(label_file, converted_annotations)
                            objects_count += len(converted_annotations)
                        else:
                            # Fallback: write as string
                            with open(label_file, 'w') as f:
                                f.write(str(label_data))
                else:
                    # Check if label file exists in dataset
                    label_path = dataset_dir / (img_path.stem + '.txt')
                    if not label_path.exists():
                        # Try to find label in labels directory
                        for labels_dir_path in dataset_dir.rglob('labels'):
                            potential_label = labels_dir_path / (img_path.stem + '.txt')
                            if potential_label.exists():
                                label_path = potential_label
                                break
                    
                    if label_path.exists():
                        annotations = parse_yolo_label(label_path)
                        if annotations:
                            converted_annotations = []
                            for ann in annotations:
                                old_class_id, x_center, y_center, width, height = ann
                                new_class_id = convert_class_id(old_class_id)
                                converted_annotations.append((new_class_id, x_center, y_center, width, height))
                            write_yolo_label(label_file, converted_annotations)
                            objects_count += len(converted_annotations)
                
                copied += 1
                if copied % 100 == 0:
                    print(f"  Processed {copied} images ({objects_count} objects)...")
            else:
                print(f"  Warning: Image not found: {img_path}")
        
        except Exception as e:
            print(f"  Error processing row {idx}: {e}")
            continue
    
    print(f"  Copied {copied} images with {objects_count} objects to {split_name} set")


def parse_yolo_label(label_file_path):
    """
    Parse YOLO format label file and return list of annotations.
    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    
    Args:
        label_file_path: Path to YOLO label file
        
    Returns:
        List of tuples: [(class_id, x_center, y_center, width, height), ...]
    """
    annotations = []
    try:
        with open(label_file_path, 'r') as f:
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
                    annotations.append((class_id, x_center, y_center, width, height))
    except Exception as e:
        print(f"  Warning: Error parsing label file {label_file_path}: {e}")
    return annotations


def convert_class_id(old_class_id, class_name_mapping=None):
    """
    Convert old class ID to our class mapping (pill=0, capsule=1).
    
    Args:
        old_class_id: Original class ID or class name
        class_name_mapping: Optional dict mapping old IDs to class names
        
    Returns:
        New class ID (0 for pill, 1 for capsule)
    """
    # If it's already in our mapping, use it
    if old_class_id in CLASS_MAPPING:
        return CLASS_MAPPING[old_class_id]
    
    # Try to convert to string and check
    old_id_str = str(old_class_id).lower().strip()
    if old_id_str in CLASS_MAPPING:
        return CLASS_MAPPING[old_id_str]
    
    # If class_name_mapping is provided, try to get class name
    if class_name_mapping and old_class_id in class_name_mapping:
        class_name = str(class_name_mapping[old_class_id]).lower().strip()
        if class_name in CLASS_MAPPING:
            return CLASS_MAPPING[class_name]
    
    # Default: assume it's a pill (class 0) if we can't determine
    print(f"  Warning: Unknown class ID '{old_class_id}', defaulting to pill (0)")
    return 0


def write_yolo_label(label_file_path, annotations):
    """
    Write YOLO format label file.
    
    Args:
        label_file_path: Path to output label file
        annotations: List of tuples [(class_id, x_center, y_center, width, height), ...]
    """
    with open(label_file_path, 'w') as f:
        for class_id, x_center, y_center, width, height in annotations:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def organize_images_and_labels(dataset_dir):
    """
    Organize images and labels directly from dataset directory structure.
    Handles YOLO format labels and maps class IDs to pill (0) and capsule (1).
    
    Args:
        dataset_dir: Path to the downloaded dataset directory
    """
    # Look for images and labels directories
    images_found = list(dataset_dir.rglob('*.jpg')) + list(dataset_dir.rglob('*.jpeg')) + \
                   list(dataset_dir.rglob('*.png')) + list(dataset_dir.rglob('*.bmp'))
    labels_found = list(dataset_dir.rglob('*.txt'))
    
    print(f"\nFound {len(images_found)} images and {len(labels_found)} label files")
    
    # Look for class names file or mapping
    class_names_file = None
    for names_file in dataset_dir.rglob('*.names'):
        class_names_file = names_file
        break
    
    class_name_mapping = {}
    if class_names_file:
        print(f"\nFound class names file: {class_names_file.name}")
        try:
            with open(class_names_file, 'r') as f:
                for idx, line in enumerate(f):
                    class_name = line.strip().lower()
                    class_name_mapping[idx] = class_name
            print(f"  Class mapping: {class_name_mapping}")
        except Exception as e:
            print(f"  Warning: Could not read class names file: {e}")
    
    if images_found:
        # Split images into train/val/test
        train_imgs, temp_imgs = train_test_split(images_found, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)
        
        # Create mapping of image names to labels
        label_map = {}
        for label_file in labels_found:
            img_name = label_file.stem
            label_map[img_name] = label_file
        
        # Statistics
        stats = {'train': {'images': 0, 'labels': 0, 'pills': 0, 'capsules': 0},
                 'val': {'images': 0, 'labels': 0, 'pills': 0, 'capsules': 0},
                 'test': {'images': 0, 'labels': 0, 'pills': 0, 'capsules': 0}}
        
        # Copy images and labels
        for split_name, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            images_dir = Path('data') / 'images' / split_name
            labels_dir = Path('data') / 'labels' / split_name
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in imgs:
                # Copy image
                dest_img = images_dir / img_path.name
                shutil.copy2(img_path, dest_img)
                stats[split_name]['images'] += 1
                
                # Process corresponding label if exists
                img_name = img_path.stem
                if img_name in label_map:
                    source_label = label_map[img_name]
                    dest_label = labels_dir / (img_name + '.txt')
                    
                    # Parse and convert labels
                    annotations = parse_yolo_label(source_label)
                    if annotations:
                        # Convert class IDs to our mapping
                        converted_annotations = []
                        for ann in annotations:
                            old_class_id, x_center, y_center, width, height = ann
                            new_class_id = convert_class_id(old_class_id, class_name_mapping)
                            converted_annotations.append((new_class_id, x_center, y_center, width, height))
                            
                            # Track class statistics
                            if new_class_id == 0:
                                stats[split_name]['pills'] += 1
                            elif new_class_id == 1:
                                stats[split_name]['capsules'] += 1
                        
                        # Write converted label
                        write_yolo_label(dest_label, converted_annotations)
                        stats[split_name]['labels'] += 1
                    else:
                        # Copy as-is if parsing failed
                        shutil.copy2(source_label, dest_label)
                        stats[split_name]['labels'] += 1
            
            total_objects = stats[split_name]['pills'] + stats[split_name]['capsules']
            print(f"  {split_name}: {stats[split_name]['images']} images, "
                  f"{stats[split_name]['labels']} labels, "
                  f"{total_objects} objects ({stats[split_name]['pills']} pills, {stats[split_name]['capsules']} capsules)")
    
    # Update data.yaml
    update_data_yaml()
    
    print("\nDataset organized successfully!")
    total_pills = sum(s['pills'] for s in stats.values())
    total_capsules = sum(s['capsules'] for s in stats.values())
    print(f"\nTotal objects labeled:")
    print(f"  Pills (class 0): {total_pills} objects")
    print(f"  Capsules (class 1): {total_capsules} objects")
    print(f"  Total: {total_pills + total_capsules} objects")


def download_sample_images():
    print("Creating synthetic pill images for testing...")
    create_synthetic_images()


def create_synthetic_images():
    import cv2
    import numpy as np
    
    raw_dir = Path('data/raw')
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating synthetic pill images for testing...")
    
    for i in range(3):
        img = np.ones((640, 640, 3), dtype=np.uint8) * 255
        
        center = (320, 320)
        
        if i % 2 == 0:
            cv2.circle(img, center, 80, (200, 200, 255), -1)
            cv2.circle(img, center, 80, (100, 100, 200), 3)
            cv2.putText(img, 'ABC', (center[0]-30, center[1]+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 150), 2)
        else:
            cv2.ellipse(img, (center[0]-40, center[1]), (40, 60), 0, 0, 360, (255, 200, 200), -1)
            cv2.ellipse(img, (center[0]+40, center[1]), (40, 60), 0, 0, 360, (255, 200, 200), -1)
            cv2.rectangle(img, (center[0]-40, center[1]-60), (center[0]+40, center[1]+60), (255, 200, 200), -1)
            cv2.ellipse(img, (center[0]-40, center[1]), (40, 60), 0, 0, 360, (200, 100, 100), 2)
            cv2.ellipse(img, (center[0]+40, center[1]), (40, 60), 0, 0, 360, (200, 100, 100), 2)
        
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        save_path = raw_dir / f"pill_sample_{i+1}.jpg"
        cv2.imwrite(str(save_path), img)
        print(f"Created: {save_path.name}")
    
    print(f"\nCreated 3 synthetic images in {raw_dir}")


if __name__ == '__main__':
    download_sample_images()
