import os
import cv2
from ultralytics import YOLO
import yaml

def get_results_dir(image_path):
    """Get the results directory for a given R1 crop"""
    # Extract original file name from the crop name
    # Format is page{idx}_crop{i}_{j}.png
    page_num = image_path.split('_')[0].replace('page', '')
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Get parent directory name from any r1 crop
    parent_dir = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
    
    # Construct results directory path
    return os.path.join('results', parent_dir)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_confidence_threshold():
    config = load_config()
    return config.get('confidence_threshold', 0.5)

def process_r2(input_paths):
    """
    Stage 2: Process R1 crops to detect and organize sub-components
    """
    output_dir = 'Falcon_r3_input'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model
    model = YOLO('models/falcon_r2.pt')
    confidence = load_confidence_threshold()
    config = load_config()
    min_crop_size = config.get('min_crop_size', 10)
    iou_threshold = config.get('iou_threshold', 0.5)
    max_detections = config.get('max_detections', 1000)
    
    processed_data = []
    
    # Process each image from R1
    for image_path in input_paths:
        print(f"Processing R2 for: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue
        
        print(f"  Image shape: {image.shape}")
        
        # Get results directory
        results_dir = get_results_dir(image_path)
        r2_crops_dir = os.path.join(results_dir, 'r2_crops')
        
        # Create folders
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, base_name)
        results_output_dir = os.path.join(r2_crops_dir, base_name)
        
        os.makedirs(image_output_dir, exist_ok=True)
        os.makedirs(results_output_dir, exist_ok=True)
        
        # Run YOLO inference
        results = model(image, conf=confidence, iou=iou_threshold, max_det=max_detections)
        
        print(f"  R2 detections found: {sum(len(r.boxes) for r in results)}")
        
        # Process detections
        detections_saved = 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get class
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # Create class directories
                class_dir = os.path.join(image_output_dir, class_name)
                results_class_dir = os.path.join(results_output_dir, class_name)
                
                os.makedirs(class_dir, exist_ok=True)
                os.makedirs(results_class_dir, exist_ok=True)
                
                # Get coordinates and crop
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Validate coordinates are within image bounds
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                # Ensure we have a valid crop region
                if x2 <= x1 or y2 <= y1:
                    print(f"Warning: Invalid crop coordinates ({x1},{y1},{x2},{y2}), skipping")
                    continue
                
                # Check minimum crop size
                crop_width = x2 - x1
                crop_height = y2 - y1
                if crop_width < min_crop_size or crop_height < min_crop_size:
                    print(f"Warning: Crop too small ({crop_width}x{crop_height}), minimum is {min_crop_size}x{min_crop_size}, skipping")
                    continue
                
                crop = image[y1:y2, x1:x2]
                
                # Validate crop is not empty
                if crop.size == 0:
                    print(f"Warning: Empty crop for coordinates ({x1},{y1},{x2},{y2}), skipping")
                    continue
                
                # Generate crop filename
                crop_name = f'{base_name}_{class_name}_{len(processed_data)}.png'
                
                # Save to processing directory
                output_path = os.path.join(class_dir, crop_name)
                cv2.imwrite(output_path, crop)
                
                # Save to results directory
                results_path = os.path.join(results_class_dir, crop_name)
                cv2.imwrite(results_path, crop)
                
                # Store processing info
                processed_data.append({
                    'path': output_path,
                    'class': class_name,
                    'original_path': image_path
                })
                
                detections_saved += 1
        
        print(f"  R2 crops saved: {detections_saved}")
    
    print(f"Total R2 processed items: {len(processed_data)}")
    
    return processed_data
