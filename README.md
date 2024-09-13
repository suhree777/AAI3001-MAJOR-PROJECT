# AAI3001-MAJOR-PROJECT
This project implements a U-Net model for tissue cell segmentation using PyTorch. It focuses on Multi-Organ Nucleus Segmentation (Monuseg), training the model with annotated images, and evaluating the performance using metrics like IoU (Intersection over Union) and surface distance.

## Project Structure:
Main Project/

├── dataset/
│   ├── train_dataset/

│   │   ├── Tissue Images/

│   │   ├── Annotations/

│   ├── test_dataset/

│   │   ├── Tissue Images/

│   │   ├── Annotations/

├── utils/

│   ├── dataset.py

│   ├── eval_metrics.py

├── models/

│   ├── u_net.py

├── train_val.py

├── test.py

└── trained_unet.pth

## Dataset
The dataset contains tissue images in .tif format and corresponding annotations in .xml format.
The dataset is split into training, validation, and test sets.

## Requirements
Python 3.x
PyTorch
Torchvision
Scikit-learn
Matplotlib
PIL

## Training
To train the model on the dataset, you can run the train_val.py script:

python train_val.py

This script will:

Load the dataset, split it into training and validation sets, and create DataLoader objects for each.
Train the U-Net model for segmentation.
Validate the model performance using IoU.
Plot the training/validation loss and IoU.
Save the trained model as trained_unet.pth

