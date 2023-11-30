import os
import shutil
from sklearn.model_selection import train_test_split

def move_files(file_list, source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    for file in file_list:
        shutil.copy(os.path.join(source_folder, file), os.path.join(destination_folder, file))

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
train_files, test_val_files = train_test_split(image_files, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)

# Further split the test set to reserve some tissue types
val_files, test_files = train_test_split(test_val_files, test_size=0.5, random_state=42)

# Move files to their respective folders
move_files(train_files, dataset_folder, train_folder)
move_files(val_files, dataset_folder, val_folder)
move_files(test_files, dataset_folder, test_folder)
