import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

CLASS_DICT = {
    0: 'paragraph', 
    1: 'text_box', 
    2: 'image', 
    3: 'table'
    }
COLORS = {
    0: (255, 0, 0), 
    1: (0, 255, 0), 
    2: (0, 0, 255), 
    3: (0, 255, 255)
    }

def draw_mask(image, masks_poly, _class):

    """
    mask draw
    
    """
    image = Image.fromarray(image)
    for x_y, _cls in zip(masks_poly, _class):
        color = COLORS[int(_cls)]
        poly = Image.new('RGBA', image.size)
        pdraw = ImageDraw.Draw(poly)
        pdraw.polygon(x_y, fill=(color[0], color[1], color[2], 70), outline=(0,0,0,255))
        image.paste(poly,mask=poly)
    # return image
    return np.asarray(image)


def draw_bbox(img, prediction_data):

    """
    draw bbox
    """

    for c, b, cl in zip(prediction_data["confidence"], \
                        prediction_data["bboxs"], prediction_data["class"]):
        bbox = [int(i) for i in b]
        class_ = CLASS_DICT[int(cl)]
        cv2.putText(img, class_+":"+str(round(c, 2)), (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, \
                        COLORS[int(cl)], 1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (bbox[0], bbox[1]),(bbox[2], bbox[3]),  COLORS[int(cl)], 1)
    return img
