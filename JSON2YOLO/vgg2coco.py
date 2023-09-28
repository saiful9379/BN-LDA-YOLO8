# -*- coding: utf8 -*-
import json
import cv2
import os

DEBUG = True

data_type_list = ["training", "validation"]

data_type = data_type_list[1]

path = f"/media/sayan/hdd1/PROJECT/MRZ_DETECTION/training/dataset/vgg_annotation/{data_type}.json"
image_dir = f"/media/sayan/hdd1/PROJECT/MRZ_DETECTION/training/dataset/images/{data_type}" 

################### output file ##########################333

OUTPUT_TXT_FILE = f"/media/sayan/hdd1/PROJECT/MRZ_DETECTION/training/dataset/{data_type}.txt"
OUTPUT_JSON_FILE = f"/media/sayan/hdd1/PROJECT/MRZ_DETECTION/training/dataset/annotations/{data_type}.json"

txt_file_path = f"./images/{data_type}"


classes_names=["MRZ"]

json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

categories = {}
with open(OUTPUT_TXT_FILE,"w") as txt_f:
    
    with open(path,"r") as json_file:
        data = json.load(json_file)
        # print(data)
        image_id = 0
        ano_id = 0
        categories_list=[]
        annotaion_list = []
        for i in data:
            if len(data[i]['regions'])!=0:

                file_name = data[i]['filename']
                print(file_name)
                img_path = os.path.join(image_dir, file_name)
                print(img_path)

                if os.path.exists(img_path):
                    txt_f.write(txt_file_path+"/"+file_name+"\n")
                    print(img_path)
                    img = cv2.imread(img_path)
                    # print(img)
                    h,w,_= img.shape


                    
                    image = {
                    "file_name": file_name,
                    "height": h,
                    "width": w,
                    "id": image_id,
                    }
                    json_dict["images"].append(image)

                    for region in data[i]['regions']:
                        class_name = region['region_attributes']['layout']
                        indexs = classes_names.index(class_name)
                        id_name = (indexs,class_name)
                        if id_name not in categories_list:
                            categories_list.append(id_name)
                        if region['shape_attributes']['name'] =='rect':
                            x,y,w,h = region['shape_attributes']['x'],region['shape_attributes']['y'],region['shape_attributes']['width'],region['shape_attributes']['height']
                        elif region['shape_attributes']['name'] =='polygon':
                            all_x,all_y = region['shape_attributes']['all_points_x'],region['shape_attributes']['all_points_y']
                            x = min(all_x)
                            y = min(all_y)
                            w = max(all_x)
                            h = max(all_y)  
                        else:
                            pass
                        
                        ann = {
                            "area": w * h,
                            "iscrowd": 0,
                            "image_id": image_id,
                            "bbox": [x, y, w, h],
                            "category_id": indexs,
                            "id": ano_id,
                            "ignore": 0,
                            "segmentation": [],
                        }
                        json_dict["annotations"].append(ann)
                        ano_id+=1
                    image_id += 1
            
            
        print("classes :",categories_list)
        # exit()
        list_of_cat= []
        categories_list = sorted([(i[0],i[1]) for i in categories_list])
        print(categories_list)
        # exit()
        # categories_list= categories_list.sort(key=lambda x: float(x[0]))
        # sorted_categoris= categories_list.sort(key=lambda x: int(x[0]))
        for i in categories_list:
            categoris_= {
            "supercategory": "none",
            "name": i[1],
            "id": i[0]       
            }
            list_of_cat.append(categoris_)
            
        json_dict['categories']=list_of_cat

with open(OUTPUT_JSON_FILE, 'w', encoding='utf8') as json_file:
    json.dump(json_dict, json_file, ensure_ascii=False)