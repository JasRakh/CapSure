from pathlib import Path
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


def download_and_prepare_dataset(dataset_id="tommyngx/epillid-data-v1"):
    """
    Download dataset from Kaggle and prepare it for YOLOv11 training.
    
    Args:
        dataset_id: Kaggle dataset identifier
    """
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    
    print(f"Loading dataset: {dataset_id}")
    
    # First, download the dataset to get the path
    print("Downloading dataset files...")
    dataset_path = kagglehub.dataset_download(dataset_id)
    print(f"Dataset downloaded to: {dataset_path}")
    
    # Try to load CSV files that might contain image paths and annotations
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


def prepare_yolo_dataset_from_dataframe(df, dataset_dir):
    """
    Prepare YOLO dataset from pandas DataFrame.
    
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
    
    # Common column name patterns
    for col in df.columns:
        col_lower = col.lower()
        if 'image' in col_lower or 'path' in col_lower or 'file' in col_lower:
            image_col = col
        if 'label' in col_lower or 'annotation' in col_lower or 'bbox' in col_lower:
            label_col = col
    
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


def process_split(df, dataset_dir, data_dir, split_name, image_col, label_col):
    """
    Process a single split (train/val/test) of the dataset.
    
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
                if label_col and label_col in row:
                    label_data = row[label_col]
                    if pd.notna(label_data):
                        # Create label file
                        label_file = labels_dir / (img_path.stem + '.txt')
                        # If label_data is a string, try to parse it
                        # This is a simplified version - you may need to adjust based on your label format
                        with open(label_file, 'w') as f:
                            f.write(str(label_data))
                
                copied += 1
                if copied % 100 == 0:
                    print(f"  Processed {copied} images...")
            else:
                print(f"  Warning: Image not found: {img_path}")
        
        except Exception as e:
            print(f"  Error processing row {idx}: {e}")
            continue
    
    print(f"  Copied {copied} images to {split_name} set")


def organize_images_and_labels(dataset_dir):
    """
    Organize images and labels directly from dataset directory structure.
    
    Args:
        dataset_dir: Path to the downloaded dataset directory
    """
    # Look for images and labels directories
    images_found = list(dataset_dir.rglob('*.jpg')) + list(dataset_dir.rglob('*.jpeg')) + \
                   list(dataset_dir.rglob('*.png')) + list(dataset_dir.rglob('*.bmp'))
    labels_found = list(dataset_dir.rglob('*.txt'))
    
    print(f"\nFound {len(images_found)} images and {len(labels_found)} label files")
    
    if images_found:
        # Split images into train/val/test
        train_imgs, temp_imgs = train_test_split(images_found, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)
        
        # Create mapping of image names to labels
        label_map = {}
        for label_file in labels_found:
            img_name = label_file.stem
            label_map[img_name] = label_file
        
        # Copy images and labels
        for split_name, imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            images_dir = Path('data') / 'images' / split_name
            labels_dir = Path('data') / 'labels' / split_name
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in imgs:
                # Copy image
                shutil.copy2(img_path, images_dir / img_path.name)
                
                # Copy corresponding label if exists
                img_name = img_path.stem
                if img_name in label_map:
                    shutil.copy2(label_map[img_name], labels_dir / (img_name + '.txt'))
            
            print(f"  {split_name}: {len(imgs)} images")
    
    print("\nDataset organized successfully!")


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
