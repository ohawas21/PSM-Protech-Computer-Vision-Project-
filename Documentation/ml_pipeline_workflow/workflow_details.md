# ML Pipeline Overview

This project builds an end-to-end machine learning pipeline to evaluate technical blueprints by extracting dimensions, tolerances, and annotations—and then classifying them based on production feasibility. This README provides an overview of the pipeline's architecture and the sequential steps involved, as represented in the accompanying UML diagram (`ml_pipeline.puml`).

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Workflow Steps](#pipeline-workflow-steps)
  - [1. Data Ingestion and Parsing](#1-data-ingestion-and-parsing)
  - [2. Blueprint Extraction](#2-blueprint-extraction)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Feature Engineering](#4-feature-engineering)
  - [5. Modeling](#5-modeling)
  - [6. Deployment and Monitoring](#6-deployment-and-monitoring)
  - [7. Pipeline Orchestration](#7-pipeline-orchestration)
- [Why the Steps Are Organized This Way](#why-the-steps-are-organized-this-way)
- [Conclusion](#conclusion)

---

## Overview

The ML pipeline is designed to process complex blueprint data in a sequential and modular manner. The process converts raw blueprint files into structured data, cleans and normalizes that data, extracts meaningful features, and ultimately utilizes multiple machine learning models to determine whether the specified tolerances are achievable in production. The key modules are:

- **Data Layer:** For file ingestion and parsing.
- **Extraction Module:** Converts blueprint images into structured, domain-specific data.
- **Data Processing and Preprocessing:** Cleans and normalizes the extracted data.
- **Feature Engineering:** Transforms normalized data into feature vectors.
- **Modeling:** Comprises several models:
  - **YOLO Detection Model:** Identifies key regions (bounding boxes) in blueprints.
  - **Symbol Classification Model:** Classifies symbols (e.g., arrows, icons) within cropped regions.
  - **OCR Model:** Extracts text from cropped regions.
  - **Rule-Based Feasibility Model:** Integrates model outputs to perform a feasibility analysis.
- **Deployment and Monitoring:** Packages and deploys the final model and monitors its performance.
- **Pipeline Orchestration:** Manages the sequential flow of tasks across the pipeline.

---

## Pipeline Workflow Steps

### 1. Data Ingestion and Parsing

- **DataIngestion**  
  - **Method:** `ingestData(sourcePath: str) → RawData`  
  - **Function:** Reads raw blueprint files (PDFs, CAD files) from specified sources.
  
- **FileParser** (Base Class)  
  - **Method:** `parse(filePath: str) → RawData`  
  - **Function:** Provides an interface for parsing files.
  
- **PDFParser & CADParser** (Inherit from FileParser)  
  - **Methods:** `parsePDF(filePath: str) → RawData` and `parseCAD(filePath: str) → RawData`  
  - **Function:** Handle format-specific parsing logic.

---

### 2. Blueprint Extraction

- **BlueprintExtractionModel**  
  - **Method:** `extractBlueprintData(filePath: str) → StructuredData`  
  - **Function:** Uses computer vision techniques (with OCR assistance) to extract dimensions, tolerances, and annotations from blueprints.
  
- **OCRProcessor** (Support Class)  
  - **Method:** `processImage(imagePath: str) → TextData`  
  - **Function:** Provides basic OCR capabilities to assist in text extraction during blueprint extraction.

---

### 3. Data Preprocessing

- **DataPreprocessing**  
  - **Method:** `cleanData(rawData: RawData) → CleanData`  
  - **Function:** Removes noise and corrects errors in the structured data.
  
  - **Method:** `normalizeData(cleanData: CleanData) → NormalizedData`  
  - **Function:** Standardizes units, formats, and scales data for consistent further processing.

---

### 4. Feature Engineering

- **FeatureEngineering**  
  - **Method:** `extractFeatures(data: NormalizedData) → FeatureVector`  
  - **Function:** Converts normalized data into a feature vector that captures essential parameters.
  
  - **Method:** `aggregateFeatures(features: FeatureVector) → AggregatedData`  
  - **Function:** Aggregates individual features into summary statistics or higher-level representations.

---

### 5. Modeling

#### 5A. Object Detection (YOLO)

- **YOLODetectionModel**  
  - **Method:** `trainYOLO(features: FeatureVector) → Model`  
  - **Function:** Trains the YOLO model to detect key regions (bounding boxes) in blueprint images.
  
  - **Method:** `predictYOLO(features: FeatureVector) → BoundingBoxes`  
  - **Function:** Predicts bounding boxes to identify regions of interest.

#### 5B. Symbol Classification

- **SymbolClassificationModel**  
  - **Method:** `trainSymbolClassifier(croppedImage: Image) → Model`  
  - **Function:** Trains a model to classify symbols in cropped image regions.
  
  - **Method:** `predictSymbols(croppedImage: Image) → Prediction`  
  - **Function:** Classifies symbols (e.g., arrows, icons) based on the cropped input.

#### 5C. OCR for Text Extraction

- **OCRModel**  
  - **Method:** `trainOCR(imageData: ImageData) → Model`  
  - **Function:** Trains an OCR model specifically tuned to your blueprint data.
  
  - **Method:** `extractText(image: Image) → TextData`  
  - **Function:** Extracts precise textual information from cropped regions.

#### 5D. Rule-Based Feasibility Analysis

- **RuleBasedFeasibilityModel**  
  - **Method:** `applyRules(yoloOutput: BoundingBoxes, symbolOutput: Prediction, ocrOutput: TextData) → FeasibilityResult`  
  - **Function:** Integrates outputs from YOLO, symbol classification, and OCR models to determine if production tolerances are feasible.

---

### 6. Deployment and Monitoring

- **ModelDeployment**  
  - **Method:** `packageModel(model: Model) → DeploymentPackage`  
  - **Function:** Packages the final feasibility model for deployment.
  
  - **Method:** `deployModel(deploymentPackage: DeploymentPackage) → Endpoint`  
  - **Function:** Deploys the model (e.g., as a REST API) for real-time usage.

- **Monitoring**  
  - **Method:** `monitorPerformance(endpoint: Endpoint) → PerformanceMetrics`  
  - **Function:** Continuously monitors the model's performance in production.
  
  - **Method:** `triggerRetraining() → None`  
  - **Function:** Initiates retraining if performance degrades.

---

### 7. Pipeline Orchestration

- **PipelineManager**  
  - **Method:** `runPipeline() → None`  
  - **Function:** Orchestrates the entire workflow—from data ingestion, extraction, preprocessing, feature engineering, and modeling, to deployment and monitoring—ensuring each step is executed in the correct sequence.

---

## Why the Steps Are Organized This Way

- **Entry Point and Standardization:**  
  *Data ingestion and file parsing* are the entry points, ensuring all raw data is collected and converted to a consistent format.

- **Specialized Extraction:**  
  The *blueprint extraction* step converts visual, unstructured data into meaningful structured data. This is critical because subsequent processing relies on having accurate, domain-specific data.

- **Cleaning and Normalization:**  
  *Data preprocessing* refines the extracted data by cleaning and normalizing it, so that the *feature engineering* step can effectively generate numerical representations.

- **Feature Engineering for ML Models:**  
  *Feature engineering* transforms cleaned data into input vectors, which are necessary for training the object detection, symbol classification, and OCR models.

- **Modeling Sequence:**  
  - **YOLODetectionModel** runs first to locate areas of interest.
  - Outputs from YOLO then drive the *symbol classification* and *OCR* tasks, ensuring that only relevant image areas are analyzed.
  - The *rule-based feasibility* step integrates the outputs to make the final production feasibility determination.

- **Deployment and Feedback:**  
  Once the decision is made, the model is packaged and deployed, and its performance is continuously monitored to ensure long-term reliability.

- **Orchestration:**  
  The *PipelineManager* oversees the entire workflow, ensuring each dependency is met and steps are executed in the correct order, maintaining the integrity and efficiency of the pipeline.

---

## Conclusion

This ML pipeline is a comprehensive, modular, and sequential process designed to address the complexities of extracting and analyzing blueprint data for production feasibility. The modular design allows individual components to be developed, tested, and maintained independently while the orchestration layer ensures smooth integration of all parts. This README, along with the accompanying UML diagram, provides a clear roadmap for both implementation and future development.

Copy and paste this README content into your markdown file to provide a detailed overview of the project workflow.
