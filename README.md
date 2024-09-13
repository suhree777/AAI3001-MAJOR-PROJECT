# AAI3001-MAJOR-PROJECT
This project implements a U-Net model for tissue cell segmentation using PyTorch. It focuses on Multi-Organ Nucleus Segmentation (Monuseg), training the model with annotated images, and evaluating the performance using metrics like IoU (Intersection over Union) and surface distance.

![TCGA-18-5592-01Z-00-DX1](https://github.com/user-attachments/assets/c7eb68c0-bfd4-4b25-81d1-6edf97db08b1)

[Monuseg.pptx](https://github.com/user-attachments/files/16991498/Monuseg.pptx)

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

```ruby
python train_val.py
```

This script will:

Load the dataset, split it into training and validation sets, and create DataLoader objects for each.
Train the U-Net model for segmentation.
Validate the model performance using IoU.
Plot the training/validation loss and IoU.
Save the trained model as trained_unet.pth

## Testing
To test the model on a separate test dataset, run the test.py script:

```ruby
python test.py
```

This script will:

Load the test dataset and perform inference using the trained U-Net model.
Calculate test loss, IoU, and surface distance metrics.
Visualize a few test predictions with their corresponding ground truth masks.

## Results
Metrics: The model’s performance is evaluated using:
IoU: Intersection over Union, a measure of overlap between the predicted and true masks.
Surface Distance: A metric used to measure how close the predicted mask boundaries are to the ground truth.
Visualization: The test script includes a visualization of the original image, predicted mask, and true mask side-by-side.


![Figure_2](https://github.com/user-attachments/assets/577b3081-6565-41b3-9ba1-e6f99787b707)


## Data Preprocessing
The images and masks are preprocessed using the MoNuSegDataset class defined in utils/dataset.py. This handles loading the images, annotations, and creating masks from the annotations.

## Model
The model architecture is a U-Net, which is commonly used for segmentation tasks. The implementation can be found in models/u_net.py.

## Key Files
train_val.py: Script for training and validating the U-Net model.
test.py: Script for testing the model on the test dataset.
u_net.py: The U-Net model implementation.
dataset.py: Dataset handling and preprocessing.
eval_metrics.py: Functions for calculating IoU and surface distance metrics.

