# Workflow Summary and Data Documentation

## Workflow So Far

The current project workflow follows a modular pipeline that includes:

1. **Synthetic Data Generation**: Using custom scripts, synthetic documents and forms were created to simulate real-world scenarios in the absence of production data.
2. **OCR Extraction Pipeline**: Text is extracted from synthetic images using an OCR engine (e.g., Tesseract or EasyOCR). This text is then cleaned and structured.
3. **Data Annotation**: Extracted OCR content is manually or semi-automatically annotated to create ground truth for model training.
4. **Classification Model Training**: A machine learning model is trained to classify extracted document content into predefined categories (e.g., invoice, order, delivery note).
5. **Evaluation and Reference Metrics**: Performance of the classification model is recorded to serve as a benchmark when real data becomes available.

## Model Used and Rationale

The classification model currently in use is a Convolutional Neural Network (CNN), chosen for the following reasons:

- CNNs perform well on visual/textual patterns common in scanned document images.
- The model architecture allows for end-to-end learning from preprocessed document images or OCR-extracted text representations.
- It can be fine-tuned or extended for multi-label classification as required.

Alternatives considered included traditional machine learning (e.g., SVM, Random Forest) which were found to underperform on noisy OCR inputs. Future iterations may explore Transformer-based architectures (e.g., BERT or LayoutLM) once more realistic datasets are available.

## Assumptions About Expected Results

The current assumptions being tested using synthetic data are:

- The synthetic documents are a close enough approximation of the real data structure to allow for meaningful model pretraining.
- The OCR engine can extract structured and semi-structured text with reasonable accuracy, especially when synthetic image quality is high.
- Classification results from synthetic data training can be used as a baseline or reference for future comparison once real data becomes available.

These assumptions will be re-evaluated after real data is acquired.

## Synthetic Data Documentation

Synthetic data was generated to simulate real business documents. Key details include:

- **Tools Used**: Python scripts using libraries like Faker for content, PIL/OpenCV for image generation, and LaTeX/HTML templates for layout.
- **Data Types**: Simulated invoices, delivery notes, and product orders, including varying layouts and fonts to introduce variability.
- **Labeling Strategy**: Each synthetic document is labeled according to its document type for supervised classification.
- **Volume**: Approximately 500–1000 synthetic documents generated to cover various business cases and formats.
- **Storage Format**: Data stored as PNG/JPG images with associated labels in CSV/JSON format for model training.

The synthetic data allows for early-stage prototyping and serves as a placeholder until production data is available under NDA.
