# from ultralytics import YOLO
import os
from ultralytics.yolo.engine.model import YOLO
# from ultralytics  import YOLO


class _config:
    model_name = "./pretrained/yolov8m-seg.pt"
    checkpoint = "./runs/segment/train28/weights/best.pt"
    model_config = "./data/yolov8.yaml"
    epoch = 300
    img_dim = 1024
    batch_size = 6
    max_det = 600
    resume=True

cfg = _config()


"""
# from checkpoint training need to use 
model = YOLO("last.pt")
model.train(resume=True)
"""

model_path = cfg.checkpoint if cfg.checkpoint else cfg.model_name
resume = True if cfg.checkpoint else False

print("Resume : ", resume)
print("CheckPoint path :" , model_path)

model = YOLO(model_path)
model.train(
    data=cfg.model_config, 
    epochs=cfg.epoch, 
    imgsz=cfg.img_dim, 
    batch=cfg.batch_size,
    max_det = cfg.max_det,
    resume = cfg.resume
    )