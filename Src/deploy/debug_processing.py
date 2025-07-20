#!/usr/bin/env python3
"""
Debug script to test the processing pipeline and identify cropping issues
"""
import os
import sys
import yaml
from processing.falcon_r1 import process_r1
from processing.falcon_r2 import process_r2

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def debug_processing(test_file_path):
    """Debug the processing pipeline with detailed output"""
    print("=" * 60)
    print("DEBUG PROCESSING PIPELINE")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print(f"Configuration loaded:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Check if test file exists
    if not os.path.exists(test_file_path):
        print(f"Error: Test file '{test_file_path}' not found")
        return
    
    print(f"Input file: {test_file_path}")
    print(f"File size: {os.path.getsize(test_file_path)} bytes")
    print()
    
    try:
        # Stage 1: R1 Processing
        print("STAGE 1: R1 Processing")
        print("-" * 30)
        r1_output = process_r1(test_file_path)
        print(f"R1 Output paths ({len(r1_output)} items):")
        for i, path in enumerate(r1_output):
            if os.path.exists(path):
                file_size = os.path.getsize(path)
                print(f"  {i+1}. {os.path.basename(path)} ({file_size} bytes)")
            else:
                print(f"  {i+1}. {os.path.basename(path)} (FILE NOT FOUND)")
        print()
        
        if not r1_output:
            print("No output from R1 processing. Check model and confidence threshold.")
            return
        
        # Stage 2: R2 Processing
        print("STAGE 2: R2 Processing")
        print("-" * 30)
        r2_output = process_r2(r1_output)
        print(f"R2 Output items ({len(r2_output)} items):")
        for i, item in enumerate(r2_output):
            path = item['path']
            class_name = item['class']
            if os.path.exists(path):
                file_size = os.path.getsize(path)
                print(f"  {i+1}. {os.path.basename(path)} (class: {class_name}, {file_size} bytes)")
            else:
                print(f"  {i+1}. {os.path.basename(path)} (class: {class_name}, FILE NOT FOUND)")
        print()
        
        # Summary
        print("PROCESSING SUMMARY")
        print("-" * 30)
        print(f"R1 Crops Generated: {len(r1_output)}")
        print(f"R2 Classifications: {len(r2_output)}")
        
        # Check results directory structure
        results_dirs = [d for d in os.listdir('results') if os.path.isdir(os.path.join('results', d))]
        print(f"Results directories: {results_dirs}")
        
        for results_dir in results_dirs:
            results_path = os.path.join('results', results_dir)
            print(f"\nContents of {results_path}:")
            for item in os.listdir(results_path):
                item_path = os.path.join(results_path, item)
                if os.path.isdir(item_path):
                    sub_items = os.listdir(item_path)
                    print(f"  {item}/ ({len(sub_items)} items)")
                else:
                    print(f"  {item}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_processing.py <path_to_test_file>")
        print("Example: python debug_processing.py test_document.pdf")
        sys.exit(1)
    
    test_file = sys.argv[1]
    debug_processing(test_file)
