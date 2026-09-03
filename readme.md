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
#


## Part 1: Gloved vs Ungloved Hand Detection(Practical Task)Scenario:

# You are building a safety compliance system that checks whether workers are wearing gloves. This will be deployed on video streams or snapshots from factory cameras.
## task is to detect:
- gloved_hand
- bare_hand


## Dataset
The dataset contains images of the target object and corresponding YOLO-format annotations.
The dataset is divided into:
- Training images
- Validation images

## data.yaml
path: /content/glove_dataset
train: images/train
val: images/val
test: images/test
names:
  0: Gloves
  1: No-Gloves


## Training parameter
- model selection (yolo)
- Epoch (300)
- Image Size (640)
- Batch Size (16)
- Optimizer (AdamW optimization Algorithm)
- Initial Learning Rate (0.001)
- Final Learning Rate (0.01)

#### Model : best2002.pt 

## Future Improvements
# Possible improvements include:
- Increasing the size of the training dataset
- Adding more object classes
- Improving annotation quality
- Hyperparameter tuning
- Training for more epochs
- Testing different YOLO model sizes
- Real-time webcam detection
- Image processing 
- Model Performace graph optimisers
- Accuraccy Monitoring
- FastApi
- Cloud STorage and loggers with Databases


## Limitations

# Detection performance depends on the quality and diversity of the training dataset.The model may perform poorly when:
- The object is heavily occluded
- Lighting conditions are very different
- The object is too small
- The camera angle is significantly different
- The object is outside the training distribution
- CCTV FPS drop
- N Number of alerts when rules are broken.
- High Inference and Data processors.