# Object Detection and Tracking

## Description
This project implements real-time object detection and tracking using YOLOv5 and OpenCV. The application captures video from a webcam, detects objects, and tracks the first detected object using the CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) tracker.

## Features
- Real-time object detection using YOLOv5.
- Tracking of the first detected object in the video stream.
- Displays bounding boxes and confidence scores for detected objects.
- User-friendly interface using OpenCV.

## Installation

### Prerequisites
- Python 3.7 or higher.
- Required libraries: OpenCV, PyTorch, and NumPy.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/adil04imran/codealpha_tasks.git

Navigate to the project directory:

cd codealpha_tasks/object_detection_tracking

install the required packages:
pip install opencv-python torch torchvision numpy

Run the application with the following command:
python main.py

## How It Works
- The application captures video frames from the webcam.
- YOLOv5 detects objects in each frame and displays bounding boxes with confidence scores.
- The first detected object is tracked using the CSRT tracker.
- The tracking information is updated in real-time, and the tracking status is displayed on the video feed.

## Customization
You can adjust the confidence threshold for object detection by modifying the `if conf > 0.5:` line in the code. You can also experiment with different YOLOv5 models by changing the model parameter in the `torch.hub.load()` function.
