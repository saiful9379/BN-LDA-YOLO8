# from ultralytics import YOLO
import os
from ultralytics.yolo.engine.model import YOLO
# from ultralytics  import YOLO

model_name = "yolov8x-seg.pt"
# model_name = "yolov8m-seg.pt"
model_config = "./data/yolov8.yaml"

checkpoint = ""

"""
# from checkpoint training need to use 
model = YOLO("last.pt")
model.train(resume=True)
"""

model_path = checkpoint if checkpoint else model_name
resume = True if checkpoint else False

print("Resume : ", resume)
print("CheckPoint path :" , model_path)

model = YOLO(model_path)
model.train(
    data=model_config, 
    epochs=600, 
    imgsz= 1024, 
    batch=4,
    max_det=600,
    resume=False,
    optimizer="AdamW"
    )