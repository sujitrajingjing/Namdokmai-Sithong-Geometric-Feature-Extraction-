# Automated Geometric Feature Extraction of Namdokmai Sithong Mangoes
This project is about using an image of a mango to find which stage of ripening the mango is on.
This project provides an automated system for geometric feature extraction of Namdokmai Sithong mangoes using YOLOv8 and Python. The framework detects mango fruits and peduncles, converts pixel measurements into centimeters, and calculates key morphological traits such as fruit length and width.

The system was developed to support agricultural quality assessment and post-harvest grading applications.

📂 Features
- Object detection with YOLOv8
- Pixel-to-centimeter conversion using calibration markers
- Automated measurement of fruit length and top/bottom widths
- Comparison with expert-annotated ground truth

📸 Dataset
Expert inspection and manual annotation were performed to define ground-truth bounding boxes for mango fruits and peduncles. The datasets were specifically designed for training and evaluating YOLOv8, chosen for its superior performance in real-time agricultural scenarios.
For dataset access, please refer to the linked Kaggle dataset
 ([Namdokmai Sithong Geometric Feature Dataset](https://www.kaggle.com/datasets/sujitraarw/namdokmai-sithong-geometric-feature-dataset)).
