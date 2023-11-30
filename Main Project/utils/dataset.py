import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MoNuSegDataset(Dataset):
    def __init__(self, image_paths, annotation_paths, mask_dir, transform=None):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.mask_dir = mask_dir
        self.transform = transform or transforms.ToTensor()
        # Ensure the mask directory exists
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        annotation_path = self.annotation_paths[index]

        if os.path.exists(annotation_path):
            annotations = self.parse_xml(annotation_path)

            # Read image
            image = cv2.imread(image_path)
            #image = Image.open(image_path).convert('RGB')

            # Visualize annotations on the image
            self.visualize_annotations(image, annotations)

            # Save the visualized image to the specified directory
            filename = os.path.basename(image_path)
            save_filename = f"visualized_{os.path.splitext(filename)[0]}_{len(annotations)}.png"
            save_path = os.path.join(self.mask_dir, save_filename)

            # Check if the save_path directory exists, if not, create it
            if not os.path.exists(self.mask_dir):
                os.makedirs(self.mask_dir)

            self.save_visualized_image(image, annotations, save_path)

            # Assuming you want to return the image and annotations for training
            return image, annotations
        else:
            print(f"Warning: Annotation file not found for image {image_path}.")


    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract bounding box coordinates from the Vertices section
        boxes = []
        for region in root.findall('.//Region'):
            vertices = region.find('.//Vertices')
            if vertices is not None:
                box = [(float(vertex.get('X')), float(vertex.get('Y'))) for vertex in vertices.findall('.//Vertex')]
                boxes.append(box)
            else:
                print("Warning: Vertices not found in XML. Region information:")
                print(f"Region XML content:\n{ET.tostring(region, encoding='utf-8').decode('utf-8')}")
                print(f"XML path: {xml_path}")

        return boxes

    def visualize_annotations(self, image, annotations):
        # Draw bounding boxes on the image
        for vertices in annotations:
            pts = np.array(vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)


    def save_visualized_image(self, image, annotations, save_path):
        # Draw bounding boxes on the image
        for vertices in annotations:
            pts = np.array(vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # Save the image with bounding boxes
        filename = os.path.basename(save_path)
        cv2.imwrite(save_path, image)

