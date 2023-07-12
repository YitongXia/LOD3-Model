import os
import json
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def get_dataset_dicts(dataset_dir, json_file):
    with open(json_file, "r") as f:
        dataset_dicts = json.load(f)

    for item in dataset_dicts:
        item["file_name"] = os.path.join(dataset_dir, item["file_name"])
        for ann in item["annotations"]:
            ann["bbox_mode"] = BoxMode.XYWH_ABS

    return dataset_dicts

if __name__ == '__main__':

    dataset_dir = "path/to/your/dataset"
    train_json = "path/to/your/train_annotations.json"
    val_json = "path/to/your/val_annotations.json"

    DatasetCatalog.register("my_dataset_train", lambda:get_dataset_dicts(dataset_dir, train_json))
    MetadataCatalog.get("my_dataset_train").set(thing_classes=["class1", "class2", "class3"])
    DatasetCatalog.register("my_dataset_val", lambda:get_dataset_dicts(dataset_dir, val_json))
    MetadataCatalog.get("my_dataset_val").set(thing_classes=["class1", "class2", "class3"])

    metadata = MetadataCatalog.get("my_dataset_train")


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = (500, 750)
    cfg.SOLVER.GAMMA = 0.1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.TEST.EVAL_PERIOD = 100

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Validation
    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)