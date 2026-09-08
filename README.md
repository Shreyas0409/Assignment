'''  object detection pipeline'''


## Overview
This project uses the Ultralytics YOLO object detection framework to train a custom model for detecting a specific object.
In this project, the model is trained to detect **gloves** in images.
The trained YOLO model can detect gloves and draw bounding boxes around the detected objects.

#### Features

- Custom YOLO object detection
- Training using a custom dataset
- Validation and evaluation of the trained model
- Image-based object detection & video
- Confidence threshold control
- Saves annotated detection results
- Trained model saved as `best.pt`
- Computed the results and logs the JSON file
- Filtering the custom models of objects.
- Multithreading for processing multiple files at a time .
- Creating bounding for valid and Invalid boxes (Green ,RED)
- 


## Technologies Used

- Python
- Google Colab
- Ultralytics YOLO
- PyTorch
- OpenCV
- NumPy


# argument     |  Description                      | field
#
 1) -Input    -> Image Input Directory             --> Required  
 2) -output   -> Output directory                  --> Required
 3) -model    -> YOLOv8 (Pretrained model)         --> best.pt
 4) -confidence -> Detection confidence threshold  -> 0.40(40%)
 5) -logs     ->Data restoring logs                --> DB


Use Case: { Gloves , No-Gloves }

Accuraccy : 80% on Avg.

<img width="640" height="640" alt="1_annotated" src="https://github.com/user-attachments/assets/e547f58b-b883-4ece-9769-11b8d797119b" />
<img width="640" height="640" alt="PP02img1040_jpg rf b5dea08ccf4bbf283adeb7bdc8b0a145" src="https://github.com/user-attachments/assets/9c12824b-4e04-4222-b392-c2604f415e99" />



