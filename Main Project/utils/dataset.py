import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MoNuSegDataset(Dataset):
    def __init__(self, image_paths, mask_dir, annotation_paths=None, transform=None):
        # Initialize the dataset with image paths, mask directory, annotation paths, and transformation
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.annotation_paths = annotation_paths
        self.transform = transform or transforms.ToTensor()

        # Ensure the mask directory exists; create it if not
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        # If annotation paths are provided, create masks
        if annotation_paths is not None:
            self._create_masks()
        
    def __len__(self):
        # Return the number of images in the dataset
        return len(self.image_paths)

    def __getitem__(self, index):
        # Get an image and its corresponding mask at the given index
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        
        # Generate the mask name based on the image name
        mask_name = os.path.basename(img_path).replace('.tif', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')

        # Apply transformations to the image and mask
        img = self.transform(img)
        mask = self.transform(mask)

        return img, mask

    def _create_masks(self):
        # Create masks for all images using annotations
        for img_path, annotation_path in zip(self.image_paths, self.annotation_paths):
            mask_name = os.path.basename(img_path).replace('.tif', '.png')
            mask_path = os.path.join(self.mask_dir, mask_name)

            # If the mask does not exist, create it
            if not os.path.exists(mask_path):
                self._create_mask(annotation_path, Image.open(img_path).size, mask_path)

    def _create_mask(self, xml_path, img_shape, mask_path):
        # Create a mask from an XML annotation file
        mask = self._xml_to_mask(xml_path, img_shape)
        mask_image = Image.fromarray(mask)
        mask_image.save(mask_path)

    def _xml_to_mask(self, xml_file, img_shape):
        # Convert XML annotation to a binary mask
        tree = ET.parse(xml_file)
        root = tree.getroot()
        mask = np.zeros(img_shape[:2], dtype=np.uint8)  # Assuming img_shape is in (H, W) format

        # Iterate over regions in the XML file and fill the corresponding polygons in the mask
        for region in root.iter('Region'):
            polygon = []
            for vertex in region.iter('Vertex'):
                x = int(float(vertex.get('X')))
                y = int(float(vertex.get('Y')))
                polygon.append((x, y))

            np_polygon = np.array([polygon], dtype=np.int32)
            cv2.fillPoly(mask, np_polygon, 128)  # Fill polygon with a value (128 in this case)

        return mask
