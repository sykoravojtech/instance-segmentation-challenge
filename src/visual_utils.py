import cv2
from detectron2.utils.visualizer import Visualizer
from utils import cv2_imshow
from dataset_utils import get_data_dicts
from detectron2.data import MetadataCatalog
import os
import random
import matplotlib.pyplot as plt


def visualize_annotation_from_dict(d, object_metadata):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=object_metadata, scale=1)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])


def visualize_n_annotations(data_dir, split="train", get_random=False, n_samples=1):
    # Load the dataset
    dataset_dicts = get_data_dicts(os.path.join(data_dir, split))

    object_metadata = MetadataCatalog.get("custom_dataset_train")

    if get_random:
        samples = random.sample(dataset_dicts, n_samples)
    else:
        samples = dataset_dicts[:n_samples]

    for d in samples:
        print(d)
        visualize_annotation_from_dict(d, object_metadata)


def print_masks(outputs):
    masks = outputs['instances'].pred_masks.cpu().numpy()
    num_masks = masks.shape[0]
    fig, axs = plt.subplots(1, num_masks, figsize=(15, 5))
    
    # print(masks.shape)
    # Handle the case where there is only one mask
    if num_masks == 1:
        axs = [axs]  # Convert the single `Axes` object into a list for consistency
    
    for i, m in enumerate(masks):
        axs[i].imshow(m, cmap='gray')
        axs[i].axis('off')
    plt.show()