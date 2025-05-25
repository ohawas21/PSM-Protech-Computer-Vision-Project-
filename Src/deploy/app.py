from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for, flash
import os
import logging
import subprocess
from falcon_r2 import check_all_models  # Import the model check function

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages
UPLOAD_FOLDER = 'falcon_r1_preprocess'
OUTPUT_FOLDER = 'output'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def cleanup_folders():
    folders_to_clean = ['falcon_r1_preprocess', 'falcon_r2_preprocess', 'falcon_r3', 'falcon_r4']
    for folder in folders_to_clean:
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove empty folder
            except Exception as e:
                logging.error(f'Error cleaning up {file_path}: {e}')
    logging.info('Cleanup completed.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.error('No file part in the request')
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        logging.error('No file selected for upload')
        return 'No selected file', 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    logging.info(f'File uploaded successfully: {file_path}')

    try:
        # Check if all required models are available
        if not check_all_models():
            flash('Some models are missing. Please check the logs for details.', 'error')
            return redirect(url_for('index'))

        # Call falcon_r1.py
        logging.info('Starting falcon_r1 preprocessing...')
        subprocess.run(['python', 'falcon_r1.py'], check=True)

        # Call falcon_r2.py
        logging.info('Starting falcon_r2 processing...')
        subprocess.run(['python', 'falcon_r2.py'], check=True)

        # Call falcon_r3.py
        logging.info('Starting falcon_r3 classification...')
        subprocess.run(['python', 'falcon_r3.py'], check=True)

        # Call falcon_r4.py
        logging.info('Starting falcon_r4 data extraction...')
        subprocess.run(['python', 'falcon_r4.py'], check=True)

        # Cleanup intermediate folders
        logging.info('Starting cleanup of intermediate folders...')
        cleanup_folders()

        flash('File uploaded and processed successfully!', 'success')
    except Exception as e:
        flash(f"An unexpected error occurred: {str(e)}", 'error')
        logging.error(f'Unexpected error: {e}')

    return redirect(url_for('index'))

@app.route('/download', methods=['GET'])
def download_file():
    excel_file = os.path.join(OUTPUT_FOLDER, 'final_output.xlsx')
    if not os.path.exists(excel_file):
        logging.error('No file available for download')
        return 'No file available for download', 404
    logging.info(f'File downloaded: {excel_file}')
    return send_file(excel_file, as_attachment=True)

@app.route('/progress', methods=['GET'])
def get_progress():
    if not os.path.exists('progress.log'):
        return jsonify({'progress': 'No progress log found.'})
    with open('progress.log', 'r') as log_file:
        logs = log_file.readlines()
    return jsonify({'progress': logs[-10:]})  # Return the last 10 log entries

@app.route('/check_models', methods=['GET'])
def check_models():
    required_models = [
        'models/falcon_r1.pt',
        'models/falcon_r2.pt',
        'models/falcon_r3.pt'  # Exclude falcon_r4.pt as it is not a model
    ]

    missing_models = [model for model in required_models if not os.path.exists(model)]

    if missing_models:
        return jsonify({
            'status': 'error',
            'message': f"Missing models: {', '.join(missing_models)}"
        })

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
