# Pothole Detection Project

A computer vision-based web application for detecting potholes in road images using deep learning and OpenCV techniques.

##  Project Overview

This project aims to assist in road safety and maintenance by detecting potholes in uploaded images using multiple models:
- **ResNet50**
- **VGG19**
- **YOLOv8**

The app is built using **Flask**, with an easy-to-use web interface and secure user authentication.

---

## Features

-  Pothole detection using state-of-the-art deep learning models
-  Upload images and view predictions side by side
-  Login and Register system using MySQL
-  Supports both Deep Learning and OpenCV-based detection
-  Responsive and modern UI with HTML/CSS and JavaScript

---

##  Models Used

| Model     | Description |
|-----------|-------------|
| ResNet50  | A residual neural network with 50 layers for feature extraction |
| VGG19     | A deep convolutional network with 19 layers |
| YOLOv8    | Real-time object detection with high accuracy |
| OpenCV    | Traditional image processing for pothole contour detection |

---

##  Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Database**: MySQL
- **Deep Learning**: TensorFlow/Keras, YOLOv8 (Ultralytics)
- **Other Tools**: OpenCV, NumPy, Matplotlib

---

##  Folder Structure
project/
│
├── static/ # CSS, JS, images
├── templates/ # HTML templates
├── models/ # Saved model files
├── train/ # Training scripts
├── test/ # Test images and evaluation
├── app.py # Main Flask application
├── database.sql # MySQL DB schema
└── README.md
---

##  How to Run the Project

1.Clone the repository
   
2.Set up virtual environment

3.Run the app
Author:
Umesh Reddy

GitHub: Umeshreddy1954

Acknowledgements:
Ultralytics YOLO

TensorFlow & Keras documentation

OpenCV community tutorials
