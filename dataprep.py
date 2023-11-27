import os
import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

def parse_xml(xml_path):
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

def visualize_annotations(image_path, annotations):
    # Read image
    image = cv2.imread(image_path)

    # Draw bounding boxes on the image
    for vertices in annotations:
        pts = np.array(vertices, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    # Display the image with bounding boxes
    plt.imshow(image[:, :, ::-1])  # Convert BGR to RGB
    plt.title(image_path)
    plt.show()

def load_monuseg_dataset(data_dir):
    image_dir = os.path.join(data_dir, 'Tissue Images')
    annotation_dir = os.path.join(data_dir, 'Annotations')

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        annotation_file = os.path.splitext(image_file)[0] + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_file)

        if os.path.exists(annotation_path):
            annotations = parse_xml(annotation_path)
            visualize_annotations(image_path, annotations)
        else:
            print(f"Warning: Annotation file not found for image {image_path}.")
            
            
# File Directory, change accordingly
data_directory = r'C:\Users\serih\Desktop\AAI3001-MAJOR-PROJECT\MoNuSeg 2018 Training Data'
load_monuseg_dataset(data_directory)
