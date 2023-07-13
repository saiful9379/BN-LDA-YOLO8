import cv2
from PIL import Image
import numpy as np
from PIL import ImageDraw
import matplotlib.pyplot as plt
from ultralytics.yolo.engine.model import YOLO

model_path = '/media/sayan/hdd1/CV/yolov8_training/runs/segment/train28/weights/best.pt'

image_path = '/media/sayan/hdd1/CV/yolov8_training/dataset_chunk_500_100/images/validation/7b709887-9513-4508-9fd0-10b067d4c531.png'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)


def get_polygon_mask(image_np, masks_poly):
    image = Image.fromarray(image_np)
    for poly_x_y in masks_poly:
        x_y = [(i[0][0], i[0][1]) for i in poly_x_y.tolist()]
        poly = Image.new('RGBA', image.size)
        pdraw = ImageDraw.Draw(poly)
        pdraw.polygon(x_y, fill=(255,0,255,70),outline=(0,0,0,255))
        image.paste(poly,mask=poly)
    return image

index = 0

masks_poly = []

for result in results:
    for j, mask in enumerate(result.masks.data):
        mask = mask.detach().cpu().numpy()* 255
        mask  = mask.astype('uint8')
        mask = cv2.resize(mask, (W, H))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Extract the contour points from the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        contour_points = largest_contour.reshape(-1, 2)
        contour_points = contour_points.reshape((-1, 1, 2))
        cv2.polylines(img, [contour_points], isClosed=True, color=(0, 255, 0), thickness=2)
        masks_poly.append(contour_points)
        index+=1

image = get_polygon_mask(img, masks_poly)
img = np.asarray(image)
cv2.imwrite(f'./logs/output_.png', img)
