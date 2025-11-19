from pathlib import Path
import os


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
