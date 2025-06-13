import os
import cv2
from ultralytics import YOLO

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

def process_r2(input_paths):
    """
    Stage 2: Process R1 crops to detect and organize sub-components
    """
    output_dir = 'Falcon_r3_input'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load YOLO model
    model = YOLO('models/falcon_r2.pt')
    
    processed_data = []
    
    # Process each image from R1
    for image_path in input_paths:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue
        
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
        results = model(image)
        
        # Process detections
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
                crop = image[y1:y2, x1:x2]
                
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
    
    return processed_data
