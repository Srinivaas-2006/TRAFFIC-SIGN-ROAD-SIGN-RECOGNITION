# Traffic Sign Recognition

## Project Description

This project focuses on developing a Traffic Sign Recognition system that can classify traffic sign images into their respective categories.

The objective of this project is to recognize traffic signs from real-world images using image processing, machine learning, and deep learning techniques.

An initial classification model was developed using the Support Vector Machine algorithm. After that, dataset cleaning, image preprocessing, and CNN architecture design were completed to improve the project and prepare it for deep learning-based classification.

## Dataset Used

**GTSRB - German Traffic Sign Recognition Benchmark**

The GTSRB dataset contains real-world traffic sign images belonging to multiple traffic sign categories. It is widely used for traffic sign recognition and classification tasks.

## Dataset Details

* Dataset Name: GTSRB
* Image Type: Real traffic sign images
* Number of Classes: 43
* Training Images: 26640
* Testing Images: 12630

## Work Completed

### Initial Classification Model

An initial traffic sign classification model was developed using the Support Vector Machine algorithm.

This helped to understand the basic classification process and provided a foundation for further model development.

### Dataset Cleaning

Dataset cleaning was performed to check the quality of the traffic sign images before model training.

The cleaning process included:

* Checking invalid labels
* Checking corrupted images
* Checking invalid image shapes
* Checking blank images
* Checking duplicate images

### Dataset Cleaning Output

**Training Dataset**

* Original images: 26640
* Clean images: 26640
* Removed images: 0

**Testing Dataset**

* Original images: 12630
* Clean images: 12630
* Removed images: 0

The dataset was found to be clean and suitable for further processing.

### Image Preprocessing

Image preprocessing was performed to prepare the images for model training.

The preprocessing steps included:

* Resizing images to a fixed size
* Converting images into tensor format
* Normalizing pixel values
* Creating DataLoader for batch-wise processing

Image preprocessing helps the model learn efficiently and improves training performance.

### CNN Architecture Design

A Convolutional Neural Network architecture was designed for traffic sign classification.

The CNN architecture includes:

* Convolution layers
* ReLU activation function
* Max pooling layers
* Fully connected layers
* Dropout layer
* Output layer for 43 traffic sign classes

The CNN model is designed to automatically learn important image features such as edges, shapes, colors, and traffic sign patterns.

## Technologies Used

* Python
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* PyTorch
* Torchvision
* Google Colab

## Project Status

The project has completed the initial classification model, dataset cleaning, image preprocessing, and CNN architecture design.

The dataset is now cleaned, preprocessed, and ready for CNN model training and performance evaluation.



Traffic Sign Recognition ML Project
