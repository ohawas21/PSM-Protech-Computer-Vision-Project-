#!/usr/bin/env python3
"""
Test script to debug the R4 processing issue
"""
import os
import sys

def test_r4_processing():
    """Test R4 processing with mock data"""
    print("=" * 60)
    print("R4 PROCESSING TEST")
    print("=" * 60)
    
    # Add current directory to path so we can import processing modules
    sys.path.insert(0, os.path.dirname(__file__))
    
    try:
        from processing.falcon_r4 import process_r4
        
        # Create mock input data similar to what R3 would produce
        mock_data = [
            {
                'path': 'test_image1.png',  # This won't exist, but should show the structure
                'class': 'falcon_r2',
                'classified_class': 'location_position',
                'original_path': 'results/test_doc/r1_crops/page0_crop0_0.png'
            },
            {
                'path': 'test_image2.png',
                'class': 'falcon_r2',
                'classified_class': 'part_number',
                'original_path': 'results/test_doc/r1_crops/page0_crop0_0.png'
            },
            {
                'path': 'test_image3.png',
                'class': 'falcon_r2',
                'classified_class': 'material_spec',
                'original_path': 'results/test_doc/r1_crops/page0_crop0_1.png'
            }
        ]
        
        print(f"Testing with {len(mock_data)} mock items:")
        for i, item in enumerate(mock_data):
            print(f"  {i+1}. Class: {item['classified_class']}, R2: {item['class']}")
        
        print("\nCalling process_r4...")
        result = process_r4(mock_data)
        
        print(f"\nR4 Result: {result}")
        print(f"Number of output files: {len(result) if result else 0}")
        
        # Check what was actually created
        outputs_dir = 'outputs'
        if os.path.exists(outputs_dir):
            files = os.listdir(outputs_dir)
            print(f"\nFiles in outputs directory:")
            for file in files:
                file_path = os.path.join(outputs_dir, file)
                size = os.path.getsize(file_path)
                print(f"  {file} ({size} bytes)")
        else:
            print(f"\nOutputs directory '{outputs_dir}' does not exist")
            
    except Exception as e:
        print(f"Error testing R4: {str(e)}")
        import traceback
        traceback.print_exc()

def check_dependencies():
    """Check if required dependencies are available"""
    print("=" * 60)
    print("DEPENDENCY CHECK")
    print("=" * 60)
    
    dependencies = [
        ('openpyxl', 'Excel file creation'),
        ('PIL', 'Image processing'),
        ('yaml', 'Configuration loading'),
        ('zipfile', 'Archive creation')
    ]
    
    for module_name, description in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {module_name} - {description}")
        except ImportError as e:
            print(f"✗ {module_name} - {description} - ERROR: {e}")

if __name__ == "__main__":
    check_dependencies()
    test_r4_processing()
