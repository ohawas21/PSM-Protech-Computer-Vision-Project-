from flask import Flask, render_template, request, send_file, jsonify
from flask_bootstrap import Bootstrap5
import os
from werkzeug.utils import secure_filename
import threading
import time

app = Flask(__name__)
bootstrap = Bootstrap5(app)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'Falcon_r1_preprocess'
app.config['OUTPUTS_DIR'] = 'outputs'  # Added OUTPUTS_DIR config
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global progress tracking
processing_status = {
    'current_stage': '',
    'progress': 0,
    'status': '',
    'error': None,
    'complete': False,
    'report_file_path': None  # Added for download link
}

# Define outputs directory for consistency
app.config['OUTPUTS_DIR'] = 'outputs'


# Ensure directories exist
REQUIRED_DIRS = [
    'Falcon_r1_preprocess',
    'Falcon_r2_preprocess',
    'Falcon_r3_input',
    'outputs',
    'models'
]

for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress')
def get_progress():
    return jsonify(processing_status)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    if file:
        # Reset progress status
        global processing_status
        processing_status.update({
            'current_stage': 'Uploading',
            'progress': 0,
            'status': 'Processing started',
            'error': None,
            'complete': False,
            'report_file_path': None # Reset report path
        })
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Start processing pipeline in background
        thread = threading.Thread(target=process_pipeline, args=(filepath,))
        thread.start()
        
        return 'File uploaded successfully', 200

def update_progress(stage, progress, status, error=None, complete=False):
    """Update the global processing status"""
    global processing_status
    processing_status.update({
        'current_stage': stage,
        'progress': progress,
        'status': status,
        'error': str(error) if error else None,
        'complete': complete
    })
    print(f"Progress: {stage} - {progress}% - {status}")

def process_pipeline(filepath):
    # Import processing modules
    from processing.falcon_r1 import process_r1
    from processing.falcon_r2 import process_r2
    from processing.falcon_r3 import process_r3
    from processing.falcon_r4 import process_r4
    
    try:
        # Stage 1: Initial processing
        update_progress('Stage 1', 0, 'Starting initial processing...')
        r1_output = process_r1(filepath)
        update_progress('Stage 1', 20, 'Initial processing complete') # Adjusted progress
        
        # Stage 2: YOLO processing and classification
        update_progress('Stage 2', 20, 'Starting YOLO processing...')
        r2_output = process_r2(r1_output)
        update_progress('Stage 2', 40, 'YOLO processing complete') # Adjusted progress
        
        # Stage 3: Classification
        update_progress('Stage 3', 40, 'Starting classification...')
        r3_output = process_r3(r2_output)
        update_progress('Stage 3', 60, 'Classification complete') # Adjusted progress
        
        # Stage 4: Data extraction and Excel generation
        update_progress('Stage 4', 60, 'Starting data extraction...')
        excel_paths = process_r4(r3_output) # process_r4 now returns a list of paths
        
        report_to_download = None
        if excel_paths and isinstance(excel_paths, list) and len(excel_paths) > 0:
            # Look for Excel files (not zip files) in the returned paths
            excel_files = [path for path in excel_paths if path.endswith('.xlsx')]
            
            if excel_files:
                # Prefer the detailed extraction results file
                for path in excel_files:
                    if "extraction_results.xlsx" in os.path.basename(path):
                        report_to_download = path
                        break
                
                # If no extraction_results file, use the first Excel file
                if not report_to_download:
                    report_to_download = excel_files[0]
                
                # Store the absolute path for reliable file serving
                processing_status['report_file_path'] = os.path.abspath(report_to_download)
                update_progress('Stage 4', 85, f'Data extraction complete. Report ready: {os.path.basename(report_to_download)}')
            else:
                # No Excel files found, check if we have any files at all
                zip_files = [path for path in excel_paths if path.endswith('.zip')]
                if zip_files:
                    update_progress('Stage 4', 85, f'Image archive created: {len(zip_files)} zip file(s). Excel report generation may have failed.')
                else:
                    update_progress('Stage 4', 85, 'No output files generated by R4.')
        
        # Cleanup
        update_progress('Cleanup', 85, 'Cleaning up temporary files...')
        cleanup_folders()
        update_progress('Cleanup', 95, 'Cleanup complete.') # Adjusted progress
        
        update_progress('Complete', 100, 'Processing complete', complete=True)
        
    except Exception as e:
        error_msg = str(e)
        app.logger.error(f"Error in processing pipeline: {error_msg}", exc_info=True) # Added app.logger and exc_info
        update_progress('Error', processing_status.get('progress',0) , 'Processing failed', error=error_msg, complete=True) # Mark complete on error too

def cleanup_folders():
    """Clean up processing folders but preserve results directory"""
    import shutil
    
    # Add results directory to required dirs
    if 'results' not in REQUIRED_DIRS:
        REQUIRED_DIRS.append('results')
    
    # Ensure OUTPUTS_DIR is also created if not part of REQUIRED_DIRS already
    if app.config['OUTPUTS_DIR'] not in REQUIRED_DIRS:
        os.makedirs(app.config['OUTPUTS_DIR'], exist_ok=True)

    print("\\nProcessing directories structure:")
    for folder in ['Falcon_r1_preprocess', 'Falcon_r2_preprocess', 'Falcon_r3_input', 'results']:
        if os.path.exists(folder):
            contents = os.listdir(folder)
            print(f"\n{folder}/")
            for item in contents:
                print(f"  - {item}")
    
    # Clean only temporary processing folders (keep export_images and results)
    for folder in ['Falcon_r1_preprocess', 'Falcon_r2_preprocess', 'Falcon_r3_input']:
        print(f"\nCleaning {folder}...")
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print(f"  Removed: {filename}")
            except Exception as e:
                print(f'  Error cleaning up {file_path}: {str(e)}')
    
    # Clean up temporary export folders (keep the zip files in outputs)
    export_base_dir = 'export_images'
    if os.path.exists(export_base_dir):
        print(f"\nCleaning temporary export directory: {export_base_dir}")
        try:
            shutil.rmtree(export_base_dir)
            print(f"  Removed: {export_base_dir}")
        except Exception as e:
            print(f'  Error cleaning up {export_base_dir}: {str(e)}')

@app.route('/download_report') # Keep existing for main page
def download_last_report():
    report_path = processing_status.get('report_file_path')
    if report_path and os.path.exists(report_path):
        try:
            abs_report_path = os.path.abspath(report_path)
            if not os.path.exists(abs_report_path):
                 app.logger.error(f"Absolute report path {abs_report_path} does not exist.")
                 return "Report file not found on server (abs path error).", 404
            file_name = os.path.basename(abs_report_path)
            return send_file(abs_report_path, as_attachment=True, download_name=file_name)
        except Exception as e:
            app.logger.error(f"Error sending file '{abs_report_path}': {e}", exc_info=True)
            return "Error downloading file.", 500
    else:
        if not report_path:
            app.logger.warning("Download request (last_report) but no report_path in status.")
        elif report_path and not os.path.exists(os.path.abspath(report_path)): # Check report_path exists before logging
            app.logger.warning(f"Download request (last_report) but report_path '{os.path.abspath(report_path)}' does not exist.")
        return "Last processed report not found, not yet ready, or an error occurred during processing.", 404

@app.route('/download_file/<filename>')
def download_specific_file(filename):
    # Sanitize filename
    safe_filename = secure_filename(filename)
    if not (safe_filename.endswith('.xlsx') or safe_filename.endswith('.zip')): # Allow both xlsx and zip
        app.logger.warning(f"Attempt to download unsupported file type via specific download: {safe_filename}")
        return "Invalid file type.", 400

    file_path = os.path.join(app.config['OUTPUTS_DIR'], safe_filename)
    abs_file_path = os.path.abspath(file_path)

    # Security check: ensure the path is within the intended outputs directory
    abs_outputs_dir = os.path.abspath(app.config['OUTPUTS_DIR'])
    if not abs_file_path.startswith(abs_outputs_dir):
        app.logger.error(f"Attempt to download file outside of outputs directory: {abs_file_path}")
        return "Invalid file path.", 400

    if os.path.exists(abs_file_path):
        try:
            return send_file(abs_file_path, as_attachment=True, download_name=safe_filename)
        except Exception as e:
            app.logger.error(f"Error sending file '{abs_file_path}': {e}", exc_info=True)
            return "Error downloading file.", 500
    else:
        app.logger.warning(f"Download request for non-existent file: {abs_file_path}")
        return "File not found.", 404

@app.route('/reports')
def reports_page():
    outputs_dir = app.config['OUTPUTS_DIR']
    report_files = []
    if os.path.exists(outputs_dir):
        try:
            # Get both xlsx and zip files
            all_files = [f for f in os.listdir(outputs_dir) 
                        if (f.endswith('.xlsx') or f.endswith('.zip')) 
                        and os.path.isfile(os.path.join(outputs_dir, f))]
            
            # Separate files by type for better organization
            excel_files = [f for f in all_files if f.endswith('.xlsx')]
            zip_files = [f for f in all_files if f.endswith('.zip')]
            
            # Combine with type information
            for f in excel_files:
                report_files.append({'name': f, 'type': 'Excel Report', 'icon': '📊'})
            for f in zip_files:
                report_files.append({'name': f, 'type': 'Image Archive', 'icon': '🖼️'})
                
        except Exception as e:
            app.logger.error(f"Error listing files in {outputs_dir}: {e}", exc_info=True)
            # Optionally, pass an error message to the template
    return render_template('download_reports.html', reports=report_files)

@app.route('/delete_report', methods=['POST'])
def delete_report():
    data = request.get_json()
    filename_to_delete = data.get('filename')

    if not filename_to_delete:
        return jsonify({'success': False, 'error': 'No filename provided'}), 400

    # Sanitize filename to prevent directory traversal
    filename_to_delete = secure_filename(filename_to_delete)
    
    # Validate file type
    if not (filename_to_delete.endswith('.xlsx') or filename_to_delete.endswith('.zip')):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    file_path_to_delete = os.path.join(app.config.get('OUTPUTS_DIR', 'outputs'), filename_to_delete)
    abs_file_path_to_delete = os.path.abspath(file_path_to_delete)

    # Basic security check: ensure the path is within the intended outputs directory
    abs_outputs_dir = os.path.abspath(app.config.get('OUTPUTS_DIR', 'outputs'))
    if not abs_file_path_to_delete.startswith(abs_outputs_dir):
        app.logger.error(f"Attempt to delete file outside of outputs directory: {abs_file_path_to_delete}")
        return jsonify({'success': False, 'error': 'Invalid file path'}), 400

    if os.path.exists(abs_file_path_to_delete):
        try:
            os.remove(abs_file_path_to_delete)
            app.logger.info(f"Successfully deleted report: {abs_file_path_to_delete}")
            
            # If the deleted file is the one in processing_status, clear it
            current_report_path = processing_status.get('report_file_path')
            if current_report_path and os.path.abspath(current_report_path) == abs_file_path_to_delete:
                processing_status['report_file_path'] = None

            return jsonify({'success': True, 'message': f'File \'{filename_to_delete}\' deleted successfully.'}), 200
        except Exception as e:
            app.logger.error(f"Error deleting file '{abs_file_path_to_delete}': {e}", exc_info=True)
            return jsonify({'success': False, 'error': f'Error deleting file: {str(e)}'}), 500
    else:
        app.logger.warning(f"Attempt to delete non-existent file: {abs_file_path_to_delete}")
        # If the file doesn't exist but was in status, clear it
        current_report_path = processing_status.get('report_file_path')
        if current_report_path and os.path.abspath(current_report_path) == abs_file_path_to_delete:
            processing_status['report_file_path'] = None
        return jsonify({'success': False, 'error': 'File not found'}), 404

if __name__ == '__main__':
    
    app.run(host='0.0.0.0',port=5001,debug=False)
