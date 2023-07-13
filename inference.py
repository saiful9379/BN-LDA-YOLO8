import os
import cv2
import torch
import glob
import matplotlib.pyplot as plt
from ultralytics.yolo.engine.model import YOLO
from utility.visulization import draw_mask, draw_bbox

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

DEBUG = True


def load_yolov8_model(model_path:str):
    model = YOLO(model_path)
    return model


def masks_processing(mask, img_dim):
    H, W = img_dim[0], img_dim[1]
    mask = mask.detach().cpu().numpy()* 255
    mask  = mask.astype('uint8')
    mask = cv2.resize(mask, (W, H))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Extract the contour points from the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2)
    contour_points = contour_points.reshape((-1, 1, 2))

    allx_ally = [(i[0][0], i[0][1]) for i in contour_points.tolist()]

    return allx_ally

def get_polygon_of_masks(masks, img_dim):
    img_dims = [img_dim]*len(masks)
    p_polygon = list(map(masks_processing, masks, img_dims))
    return p_polygon


def get_bbox_processing(result):
    p_r = [i.detach().cpu().numpy() for i in [result.boxes.conf, result.boxes.xyxy, result.boxes.cls]]
    conf, bbox_xyxy, _cls = p_r[0].tolist(), p_r[1].tolist(), p_r[2].tolist()
    return conf, bbox_xyxy, _cls


def prediction(
        model, img, file_name = "unkown.jpg", output_dir="logs"):

    H, W, _ = img.shape
    results = model(img, device=device)

    prediction_data = {
        "bboxs" : [],
        "polygons":[],
        "class":[],
        "confidence":[]
    }

    for result in results:
        # print(result)
        try:
            masks = result.masks.data
            polyon_coordinates = get_polygon_of_masks(masks, img_dim=[H, W])
        except:
            polyon_coordinates = []
        # print(polyon_coordinates)
        prediction_data["polygons"].extend(polyon_coordinates)

        conf, bbox_xyxy, _cls = get_bbox_processing(result)
        prediction_data["bboxs"].extend(bbox_xyxy), prediction_data["class"].extend(_cls)
        prediction_data["confidence"].extend(conf)

    if DEBUG:

        img = draw_mask(img, prediction_data["polygons"], prediction_data["class"])

        img = draw_bbox(img, prediction_data)

        cv2.imwrite(os.path.join(output_dir, file_name), img)

    return prediction_data


if __name__ == "__main__" :

    from tqdm import tqdm 

    model_path = '/media/sayan/hdd1/CV/BN-LDA-YOLO8/runs/best.pt'
    image_path = '/media/sayan/hdd1/CV/BN-LDA-YOLO8/image/'
    output_dir = "logs"
    os.makedirs(output_dir, exist_ok=True)

    model = load_yolov8_model(model_path)

    files = glob.glob(image_path+"/*")
    for i in tqdm(range(len(files))):
        _file = files[i]
        img = cv2.imread(_file)
        file_name = os.path.basename(_file)
        prediction_data = prediction(
            model, 
            img, 
            file_name = file_name, 
            output_dir=output_dir
            )

    
