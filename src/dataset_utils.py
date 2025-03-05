import gdown
import tarfile
import os
import shutil
import random
from pathlib import Path
import numpy as np
import cv2
from detectron2.structures import BoxMode

class DatasetUtils:
    def __init__(self):
        self.tar_path = 'dl_dataset.tar.xz'  # Path to the dataset tar file
        self.extract_path = 'dl_challenge'  # Path where the dataset will be extracted
        self.gdrive_url = 'https://drive.google.com/uc?export=download&id=11s-GLb6LZ0SCAVW6aikqImuuQEEbT_Fb'

    def download_dataset(self):
        # Use gdown with fuzzy to handle Google Drive confirmation
        gdown.download(self.gdrive_url, self.tar_path, fuzzy=True)

    def unzip_dataset(self):
        # Create the folder if it doesn't exist
        if not os.path.exists(self.extract_path):
            os.makedirs(self.extract_path)

        # Open and extract the tar.xz file
        with tarfile.open(self.tar_path, 'r:xz') as tar:
            tar.extractall(path=self.extract_path)
            print(f'Dataset extracted to {self.extract_path}')

    def copy_and_split_directory(self, src_dir=None, dest_dir=None, train_split=0.7, val_split=0.2):
        """
        Copy and split a directory into train, val, and test directories.
        """
        if src_dir is None:
            src_dir = self.extract_path

        if dest_dir is None:
            dest_dir = f"{self.extract_path}_copy"

        # Ensure the coefficients are within valid bounds and sum up to less than 1
        if not (0 < train_split < 1 and 0 < val_split < 1 and train_split + val_split < 1):
            raise ValueError("Train and validation coefficients must be between 0 and 1, and their sum must be less than 1.")

        # Define the source and target directories
        src_path = Path(src_dir)
        dest_path = Path(dest_dir)

        # Copy the entire directory structure
        if dest_path.exists():
            raise FileExistsError(f"Destination directory '{dest_path}' already exists.")
        
        shutil.copytree(src_path, dest_path)

        # Get all example folders in the copied directory
        example_folders = [f for f in dest_path.iterdir() if f.is_dir()]

        # Shuffle the examples to randomize the split
        random.shuffle(example_folders)

        # Compute the split indices
        total_examples = len(example_folders)
        train_index = int(total_examples * train_split)
        val_index = train_index + int(total_examples * val_split)

        # Create train, val, and test directories inside the copied directory
        train_dir = dest_path / 'train'
        val_dir = dest_path / 'val'
        test_dir = dest_path / 'test'

        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)

        # Move folders to train, val, or test directory based on the split
        for i, folder in enumerate(example_folders):
            if i < train_index:
                shutil.move(str(folder), train_dir / folder.name)
            elif i < val_index:
                shutil.move(str(folder), val_dir / folder.name)
            else:
                shutil.move(str(folder), test_dir / folder.name)

        print(f"Copied and split '{src_dir}' into '{dest_path}' with {train_split*100:.1f}% train, {val_split*100:.1f}% val, and {(1-train_split-val_split)*100:.1f}% test.")


def load_mask_data(mask_path):
    """
    reads the mask and processes it into the format required by Detectron2
    """
    # print("Mask start")
    mask = np.load(mask_path)  # (num_instances, height, width) boolean
    # print(f"Mask shape: {mask.shape}")
    annotations = [] # Initialize an empty list to store annotations for each instance in the image.
    for i in range(mask.shape[0]): # for all instances
        # print(f"Instance {i}")
        instance_mask = mask[i] # Select the mask for the i-th instance.
        
        # Find contours (boundaries) of the instance. This converts the binary mask into a set of points.
        contours, _ = cv2.findContours(instance_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        segmentation = [] # List to store the segmentation polygons for the instance.
        for contour in contours:  # For each contour (which represents a boundary), convert to a list of points.
            contour = contour.flatten().tolist()  # Convert the contour to a list of points (1D).
            if len(contour) > 4:  # We only consider valid polygons with at least a few points.
                segmentation.append(contour)  # Add the valid contour to the segmentation list.
        # print(f"Segmentation: {segmentation}")

        if segmentation:  # If we have valid segmentation data for this instance, add it to the annotations.
            # print("Valid segmentation data found.")
            x, y, w, h = cv2.boundingRect(instance_mask.astype(np.uint8))
            bbox = [x, y, x + w, y + h]  # Convert to x1, y1, x2, y2

            annotations.append({
                "segmentation": segmentation,  # Store the segmentation data.
                "bbox": bbox,  # Compute a bounding box around the instance.
                "bbox_mode": BoxMode.XYXY_ABS,  # Bounding box mode for Detectron2 (XYXY format).
                "category_id": 0,  # Since you don't have object names, all objects belong to one category (ID 0).
            })
    return annotations

# Define the augmentation pipeline
# transform = A.Compose([
    # A.HorizontalFlip(p=0.5),                           # 1. Flip to create variability in positioning
    # A.RandomBrightnessContrast(p=0.2),                  # 2. Adjust brightness and contrast to simulate lighting conditions
    # A.Rotate(limit=30, p=0.5),                           # 3. Rotate to add variability in orientation
    # A.RandomScale(scale_limit=(0.5, 1.5), p=0.5),        # 4. Randomly scale to simulate different object sizes
    # A.RandomSizedBBoxSafeCrop(height=256, width=256, erosion_rate=0.2, p=0.5),  # 5. Crop region of interest while maintaining small objects
    # A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.5),  # 6. Dropout to add variability by hiding random regions
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 7. Normalize as a final preparation step before converting to tensor
    # ToTensorV2(),                                         # 8. Convert to tensor to make it ready for model input
# ])

def get_data_dicts(root_dir):
    """
    This function processes your entire dataset. 
    It loops through all folders in your dataset and prepares the data in a format that Detectron2 understands:
    """
    dataset_dicts = []  # Initialize an empty list to store the full dataset information.

    # Loop through each folder in the dataset directory (each folder represents an example).
    for idx, foldername in enumerate(os.listdir(root_dir)):
        record = {}  # Initialize a dictionary to store information for the current example.

        # Paths to the image and mask for this example.
        rgb_image_path = os.path.join(root_dir, foldername, 'rgb.jpg')  # Image file (RGB).
        # print(f"Processing {rgb_image_path} {type(rgb_image_path)}")
        mask_path = os.path.join(root_dir, foldername, 'mask.npy')  # Mask file (numpy array).

        # Read the image to get its dimensions (height, width).
        image = cv2.imread(rgb_image_path)
        # print(f"Image type: {type(image)}")
        if image is None:
            print(f"Error: Image {rgb_image_path} could not be read.")
            continue
        height, width = image.shape[:2]

        # Store metadata for the current image.
        record["file_name"] = rgb_image_path  # Path to the image file.
        record["image_id"] = idx  # Unique ID for this image (use index in loop).
        record["height"] = height  # Height of the image.
        record["width"] = width  # Width of the image.

        # Call the function that processes the mask and returns annotations (segmentations and bounding boxes).
        record["annotations"] = load_mask_data(mask_path)

        # Add the record for this example to the dataset list.
        dataset_dicts.append(record)

    return dataset_dicts  # Return the full dataset as a list of records.

if __name__ == '__main__':
    dutils = DatasetUtils()
    # dutils.download_dataset()
    # dutils.unzip_dataset()

    dutils.copy_and_split_directory(
        src_dir=None, 
        dest_dir=None,
        train_split = 0.7,  # 70% for training
        val_split = 0.2    # 20% for validation and the rest for test
    )