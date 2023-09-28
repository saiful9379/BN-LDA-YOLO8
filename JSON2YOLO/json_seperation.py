import os
import json

data_path = r"C:\Users\NSL5\Desktop\mrz\dataset\dataset\validation\data.json"
image_dir = r"C:\Users\NSL5\Desktop\mrz\dataset\dataset\validation\m2"

output_dir = r"C:\Users\NSL5\Desktop\mrz\dataset\dataset\validation\m2_data.json"


new_data = {}
with open(data_path,"r") as json_file:
    data = json.load(json_file)
    for key, value in data.items():
        # print(value)
        file_name = value['filename']
        img_path = os.path.join(image_dir, file_name)
        if os.path.exists(img_path):
            print(file_name)
            new_data[key] = value

with open(output_dir, 'w', encoding='utf8') as json_file:
    json.dump(new_data, json_file, ensure_ascii=False)
