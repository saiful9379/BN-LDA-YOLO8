import os
import cv2
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np


from ultralytics.yolo.engine.model import YOLO
from utility.visulization import draw_mask, draw_bbox


def rle_encode(mask: np.ndarray) -> str:
    pixels = mask.T.flatten()
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded

    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1

    rle[1::2] = rle[1::2] - rle[:-1:2]

    return " ".join(str(x) for x in rle)


def load_yolov8_model(model_path: str):
    model = YOLO(model_path)
    return model


def resize_mask(mask, img_dim):
    H, W = img_dim[0], img_dim[1]
    mask = mask.detach().cpu().numpy() * 255
    mask = mask.astype("uint8")
    mask = cv2.resize(mask, (W, H))

    return mask


def masks_processing(mask, img_dim):
    mask = resize_mask(mask, img_dim)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2)
    contour_points = contour_points.reshape((-1, 1, 2))

    allx_ally = [(i[0][0], i[0][1]) for i in contour_points.tolist()]

    return allx_ally


def get_polygon_of_masks(masks, img_dim):
    img_dims = [img_dim] * len(masks)
    p_polygon = list(map(masks_processing, masks, img_dims))
    return p_polygon


def get_bbox_processing(result):
    p_r = [
        i.detach().cpu().numpy()
        for i in [result.boxes.conf, result.boxes.xyxy, result.boxes.cls]
    ]
    conf, bbox_xyxy, pred_classes = p_r[0].tolist(), p_r[1].tolist(), p_r[2].tolist()
    return conf, bbox_xyxy, pred_classes


def prediction(model, img, file_name="unkown.jpg", output_dir="logs"):
    H, W, _ = img.shape
    results = model(img, device=DEVICE)

    prediction_data = {
        "bboxs": [],
        "polygons": [],
        "class": [],
        "confidence": [],
        "rle_mask": [],
    }

    for result in results:
        try:
            masks = result.masks.data
            polyon_coordinates = get_polygon_of_masks(masks, img_dim=[H, W])
        # except:
        #     polyon_coordinates = []
            # continue
            # masks =  result.masks.data

            conf, bbox_xyxy, pred_classes = get_bbox_processing(result)
            prediction_data["rle_mask"] = "\n".join(
                [
                    f"{file_name[:-4]}_{str(int(cls))}, {rle_encode(resize_mask(mask, [H, W]))}"
                    for cls, mask in zip(pred_classes, masks)
                ]
            )
            prediction_data["polygons"].extend(polyon_coordinates)
            prediction_data["bboxs"].extend(bbox_xyxy),
            prediction_data["class"].extend(pred_classes)
            prediction_data["confidence"].extend(conf)

            if DEBUG:
                img = draw_mask(img, prediction_data["polygons"], prediction_data["class"])
                img = draw_bbox(img, prediction_data)
                cv2.imwrite(os.path.join(output_dir, file_name), img)
        except:

            print("Prolematic Images : ", file_name)


    return prediction_data


if __name__ == "__main__":
    from tqdm import tqdm

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEBUG = True
    MODEL_PATH = "./runs/best.pt"
    IMAGE_PATH = "/media/sayan/hdd1/CV/bangla_layout_understanding/badlad/images/test"
    LOG_DIR = "logs"

    os.makedirs(LOG_DIR, exist_ok=True)
    files = glob.glob(IMAGE_PATH + "/*")
    model = load_yolov8_model(MODEL_PATH)

    
    with open(f"submission.csv", "w") as f:
        for i in tqdm(range(len(files))):
            try:
                _file = files[i]
                img = cv2.imread(_file)
                file_name = os.path.basename(_file)
                
                prediction_data = prediction(
                    model,
                    img,
                    file_name=file_name,
                    output_dir=LOG_DIR
                )
                if prediction_data["rle_mask"]:
                    f.write(prediction_data["rle_mask"])
            except:
                continue
