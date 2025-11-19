import cv2
import numpy as np
from pathlib import Path
import os


def enhance_contrast(image, alpha=1.5, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)


def remove_background(image, method='grabcut'):
    if method == 'grabcut':
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        h, w = image.shape[:2]
        rect = (10, 10, w-20, h-20)
        
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return image * mask2[:, :, np.newaxis]
    
    elif method == 'threshold':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_and(image, image, mask=mask)
    
    return image


def resize_images(input_dir, output_dir, target_size=(640, 640), maintain_aspect=True):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in image_extensions:
            image = cv2.imread(str(img_file))
            if image is not None:
                if maintain_aspect:
                    h, w = image.shape[:2]
                    scale = min(target_size[0] / w, target_size[1] / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
                
                output_file = output_path / img_file.name
                cv2.imwrite(str(output_file), resized)
                print(f"Resized: {img_file.name}")


def batch_preprocess(input_dir, output_dir, enhance=True, remove_bg=False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in image_extensions:
            image = cv2.imread(str(img_file))
            if image is not None:
                if enhance:
                    image = apply_clahe(image)
                
                if remove_bg:
                    image = remove_background(image)
                
                output_file = output_path / img_file.name
                cv2.imwrite(str(output_file), image)
                print(f"Processed: {img_file.name}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess images for detection')
    parser.add_argument('--input', type=str, required=True, help='Input directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--enhance', action='store_true', help='Enhance contrast')
    parser.add_argument('--remove-bg', action='store_true', help='Remove background')
    parser.add_argument('--resize', type=int, nargs=2, help='Resize to width height')
    
    args = parser.parse_args()
    
    if args.resize:
        resize_images(args.input, args.output, tuple(args.resize))
    else:
        batch_preprocess(args.input, args.output, args.enhance, args.remove_bg)
