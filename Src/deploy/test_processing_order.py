#!/usr/bin/env python3
"""
Test script to verify processing order and completion
"""
import os
import time
import glob

def monitor_outputs():
    """Monitor outputs directory for file creation"""
    print("=" * 60)
    print("MONITORING OUTPUTS DIRECTORY")
    print("=" * 60)
    
    outputs_dir = 'outputs'
    if not os.path.exists(outputs_dir):
        print(f"Outputs directory '{outputs_dir}' does not exist")
        return
    
    print("Initial state:")
    initial_files = os.listdir(outputs_dir)
    for file in initial_files:
        file_path = os.path.join(outputs_dir, file)
        size = os.path.getsize(file_path)
        print(f"  {file} ({size} bytes)")
    
    print("\nWaiting for new files to be created...")
    print("(This script will monitor for 60 seconds)")
    
    start_time = time.time()
    seen_files = set(initial_files)
    
    while time.time() - start_time < 60:
        current_files = set(os.listdir(outputs_dir))
        new_files = current_files - seen_files
        
        if new_files:
            for new_file in new_files:
                file_path = os.path.join(outputs_dir, new_file)
                size = os.path.getsize(file_path)
                timestamp = time.strftime("%H:%M:%S")
                print(f"  [{timestamp}] NEW: {new_file} ({size} bytes)")
                seen_files.add(new_file)
        
        time.sleep(1)
    
    print("\nFinal state:")
    final_files = os.listdir(outputs_dir)
    for file in final_files:
        file_path = os.path.join(outputs_dir, file)
        size = os.path.getsize(file_path)
        created = "NEW" if file not in initial_files else "OLD"
        print(f"  {file} ({size} bytes) [{created}]")

def check_file_integrity():
    """Check if created files are valid"""
    print("\n" + "=" * 60)
    print("FILE INTEGRITY CHECK")
    print("=" * 60)
    
    outputs_dir = 'outputs'
    if not os.path.exists(outputs_dir):
        print("No outputs directory found")
        return
    
    # Check Excel files
    excel_files = glob.glob(os.path.join(outputs_dir, "*.xlsx"))
    print(f"Excel files found: {len(excel_files)}")
    for excel_file in excel_files:
        size = os.path.getsize(excel_file)
        print(f"  {os.path.basename(excel_file)}: {size} bytes")
        if size > 0:
            try:
                import openpyxl
                wb = openpyxl.load_workbook(excel_file)
                print(f"    Sheets: {wb.sheetnames}")
                wb.close()
            except Exception as e:
                print(f"    ERROR reading Excel: {e}")
    
    # Check ZIP files
    zip_files = glob.glob(os.path.join(outputs_dir, "*.zip"))
    print(f"\nZIP files found: {len(zip_files)}")
    for zip_file in zip_files:
        size = os.path.getsize(zip_file)
        print(f"  {os.path.basename(zip_file)}: {size} bytes")
        if size > 0:
            try:
                import zipfile
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    files_in_zip = len(zf.namelist())
                    print(f"    Files in archive: {files_in_zip}")
            except Exception as e:
                print(f"    ERROR reading ZIP: {e}")

if __name__ == "__main__":
    monitor_outputs()
    check_file_integrity()
