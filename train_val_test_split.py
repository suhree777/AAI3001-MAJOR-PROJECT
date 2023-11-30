import os
import shutil
from sklearn.model_selection import train_test_split

# Set the path to your dataset folder
dataset_folder = '/content/drive/MyDrive/MoNuSeg/MoNuSegAnnotatedImages'

# Create folders for training, validation, and testing sets
train_folder = dataset_folder + '/train_split'
val_folder =  dataset_folder + '/val_split'
test_folder =  dataset_folder + '/test_split'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get a list of all image files in the dataset folder
image_files = [f for f in os.listdir(dataset_folder) if f.endswith(".png")]

# Split the dataset into training, validation, and testing sets
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)

# Move files to their respective folders
for file in train_files:
    shutil.copy(os.path.join(dataset_folder, file), os.path.join(train_folder, file))

for file in val_files:
    shutil.copy(os.path.join(dataset_folder, file), os.path.join(val_folder, file))

for file in test_files:
    shutil.copy(os.path.join(dataset_folder, file), os.path.join(test_folder, file))
