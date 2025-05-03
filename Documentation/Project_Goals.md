# Project Goals – Feasibility Analysis: Object Detection

| **N°** | **Target Description** | **Success Criteria (Measurability)** |
|:------:|-------------------------|--------------------------------------|
| Z1     | Automate evaluation of mechanical drawings to reduce manual checks. | YOLO model detects relevant drawing area (bounding boxes). Manual check no longer required for initial analysis. |
| Z2     | Detect the rectangle that contains technical info like values, tolerances, and part IDs. | Model identifies and localizes rectangles with >90% accuracy in test set. |
| Z3     | Use OCR to extract relevant text/numbers from the detected rectangles. | OCR extracts text with >85% accuracy and maps to structured format (e.g., JSON). |
| Z4     | Identify critical tolerances and feasibility issues based on extracted data and predefined rules. | System highlights values outside manufacturing specs automatically. Flagging works on >90% of cases. |
| Z5     | Re-evaluate updated drawings automatically and highlight changes. | Updated parts are automatically compared to previous versions; changes are clearly marked. |
| Z6     | Develop a user interface for reviewing detected data and manual corrections if needed. | Web interface displays detected values and allows edits. 100% of processed files are accessible via UI. |
| Z7     | Log system decisions and create a summary report for each drawing processed. | Each processed drawing has a log file with extracted data, tolerances, and final feasibility decision. |
