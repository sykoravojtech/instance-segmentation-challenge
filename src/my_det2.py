import torch
from detectron2.modeling import build_model
from detectron2 import model_zoo
from detectron2.config import get_cfg
import os


def get_total_param_num_of_model(model):
    return sum(p.numel() for p in model.parameters())

def create_model(cfg):
    if cfg is None:
        raise ValueError("cfg is None.")
    return build_model(cfg)

def get_custom_cfg(zoo_config_file):
    # Get a default configuration object.
    cfg = get_cfg()

    # Load a pre-trained Mask R-CNN model from the Detectron2 model zoo (trained on COCO).
    cfg.merge_from_file(model_zoo.get_config_file(zoo_config_file))

    # Set the training and validation datasets.
    cfg.DATASETS.TRAIN = ("custom_dataset_train",)  # Use the training dataset we registered.
    cfg.DATASETS.TEST = ("custom_dataset_train", "custom_dataset_val",)  # Use the validation dataset we registered. detectron2 uses the TEST as validation set

    cfg.DATALOADER.NUM_WORKERS = 4  # Set the number of worker threads for loading data.

    # Load the pre-trained weights for the model (pre-trained on the COCO dataset).
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_config_file)

    cfg.SOLVER.IMS_PER_BATCH = 2  # Number of images per batch.
    cfg.SOLVER.BASE_LR = 0.0025  # Base learning rate for training.
    cfg.SOLVER.MAX_ITER = 3000  # Maximum number of iterations (epochs).
    
    cfg.SOLVER.CHECKPOINT_PERIOD = 50
    cfg.TEST.EVAL_PERIOD = 50

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Batch size for the region of interest head.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of object classes (we only have one class: "object").

    # augment
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.SIZE = [0.8, 0.8]

    cfg.OUTPUT_DIR = "./outputAUG"

    # Set the output directory for saving model checkpoints and logs.
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # force it to use CPU if no GPU is available
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu' # Force it to use CPU
        print("Using CPU")
    else:
        print("Using GPU")

    return cfg