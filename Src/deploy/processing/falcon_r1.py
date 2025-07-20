import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from ultralytics import YOLO
import sys
import subprocess
import shutil
import yaml

def setup_results_structure(filepath):
    """Create results folder structure for a new file"""
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    
    # Create main results folder and subfolders
    results_dir = os.path.join('results', base_name)
    r1_dir = os.path.join(results_dir, 'r1_crops')
    r2_dir = os.path.join(results_dir, 'r2_crops')
    excel_dir = os.path.join(results_dir, 'excel')
    
    # Create all directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(r1_dir, exist_ok=True)
    os.makedirs(r2_dir, exist_ok=True)
    os.makedirs(excel_dir, exist_ok=True)
    
    # Copy original file to results folder
    shutil.copy2(filepath, os.path.join(results_dir, os.path.basename(filepath)))
    
    return results_dir

def check_poppler():
    """Check if poppler is installed and accessible"""
    try:
        # Check if poppler-utils is in PATH
        subprocess.run(['pdftoppm', '-v'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_confidence_threshold():
    config = load_config()
    return config.get('confidence_threshold', 0.5)

def process_r1(filepath):
    """
    Stage 1: Process uploaded file (image/PDF) and run initial YOLO model
    """
    # Setup results structure
    results_dir = setup_results_structure(filepath)
    r1_crops_dir = os.path.join(results_dir, 'r1_crops')
    
    # Also save to original processing directory for pipeline compatibility
    output_dir = 'Falcon_r2_preprocess'
    os.makedirs(output_dir, exist_ok=True)
    
    confidence = load_confidence_threshold()
    config = load_config()
    min_crop_size = config.get('min_crop_size', 10)
    iou_threshold = config.get('iou_threshold', 0.5)
    max_detections = config.get('max_detections', 1000)
    
    try:
        # Determine if input is PDF or image
        if filepath.lower().endswith('.pdf'):
            # Check for poppler installation
            if not check_poppler():
                raise RuntimeError(
                    "Poppler is not installed or not in PATH. "
                    "Please install poppler:\n"
                    "- On macOS: brew install poppler\n"
                    "- On Linux: sudo apt-get install poppler-utils\n"
                    "- On Windows: Download from http://blog.alivate.com.au/poppler-windows/"
                )
            
            # Convert PDF to images at high resolution
            try:
                pages = convert_from_path(
                    filepath,
                    dpi=300,
                    poppler_path='/opt/homebrew/bin' if sys.platform == 'darwin' else None
                )
                images = [cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR) for page in pages]
            except Exception as e:
                raise RuntimeError(f"Error converting PDF: {str(e)}")
        else:
            # Load image directly
            image = cv2.imread(filepath)
            if image is None:
                raise RuntimeError(f"Unable to load image file: {filepath}")
            images = [image]
        
        # Load YOLO model
        if not os.path.exists('models/falcon_r1.pt'):
            raise RuntimeError("YOLO model file 'models/falcon_r1.pt' not found")
        
        model = YOLO('models/falcon_r1.pt')
        processed_paths = []
        
        # Process each page/image
        for idx, img in enumerate(images):
            print(f"Processing image {idx+1}/{len(images)}, shape: {img.shape}")
            results = model(img, conf=confidence, iou=iou_threshold, max_det=max_detections)
            
            detections_found = 0
            for i, r in enumerate(results):
                boxes = r.boxes
                print(f"  Found {len(boxes)} detections with confidence >= {confidence}")
                for j, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    print(f"    Detection {j}: Original coords ({x1},{y1},{x2},{y2})")
                    
                    # Validate coordinates are within image bounds
                    h, w = img.shape[:2]
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(x1+1, min(x2, w))
                    y2 = max(y1+1, min(y2, h))
                    
                    print(f"    Detection {j}: Adjusted coords ({x1},{y1},{x2},{y2}), image size ({w},{h})")
                    
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
                    
                    # Crop the detected region
                    crop = img[y1:y2, x1:x2]
                    
                    # Validate crop is not empty
                    if crop.size == 0:
                        print(f"Warning: Empty crop for coordinates ({x1},{y1},{x2},{y2}), skipping")
                        continue
                    
                    print(f"    Detection {j}: Crop shape {crop.shape}")
                    detections_found += 1
                    
                    # Generate crop filename
                    crop_name = f'page{idx}_crop{i}_{j}.png'
                    
                    # Save to results directory
                    result_path = os.path.join(r1_crops_dir, crop_name)
                    cv2.imwrite(result_path, crop)
                    
                    # Also save to processing directory
                    process_path = os.path.join(output_dir, crop_name)
                    cv2.imwrite(process_path, crop)
                    processed_paths.append(process_path)
                    
                    print(f"    Saved crop: {crop_name}")
            
            print(f"  Total valid detections for image {idx+1}: {detections_found}")
        
        print(f"Total crops generated: {len(processed_paths)}")
        if not processed_paths:
            print("Warning: No objects detected in the input file")
        
        return processed_paths
    
    except Exception as e:
        raise RuntimeError(f"Error in Stage 1 processing: {str(e)}")
