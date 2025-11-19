import cv2
import numpy as np
from pathlib import Path
import json


def draw_detections(image, boxes, class_names, conf_threshold=0.5):
    annotated = image.copy()
    
    if boxes is not None:
        for box in boxes:
            if box.conf[0] >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names[cls]
                
                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return annotated


def save_detections_json(image_path, boxes, class_names, output_path, conf_threshold=0.5):
    detections = {
        'image': str(image_path),
        'detections': []
    }
    
    if boxes is not None:
        for box in boxes:
            if box.conf[0] >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                detections['detections'].append({
                    'class': class_names[cls],
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
                })
    
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)
    
    print(f"Detections saved to {output_path}")


def preprocess_image(image_path, target_size=(640, 640)):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_w = (target_size[0] - new_w) // 2
    pad_h = (target_size[1] - new_h) // 2
    
    padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    return padded, scale, (pad_w, pad_h)


def count_detections(boxes, class_names, conf_threshold=0.5):
    counts = {name: 0 for name in class_names.values()}
    
    if boxes is not None:
        for box in boxes:
            if box.conf[0] >= conf_threshold:
                cls = int(box.cls[0])
                class_name = class_names[cls]
                counts[class_name] += 1
    
    return counts
