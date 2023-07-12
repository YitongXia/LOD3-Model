import json
import random
from pycocotools.coco import COCO

# Change this path to the path of your COCO dataset JSON file
coco_json_path = '../../import_Almere/facade-2.json'

with open(coco_json_path, 'r') as f:
    data = json.load(f)

train_ratio = 0.8  # Change this value to the desired training set ratio

image_ids = [img['id'] for img in data['images']]

random.shuffle(image_ids)

num_train = int(len(image_ids) * train_ratio)
train_ids = image_ids[:num_train]
val_ids = image_ids[num_train:]

train_data = {
    # 'info': data['info'],
    # 'licenses': data['licenses'],
    'categories': data['categories'],
    'images': [img for img in data['images'] if img['id'] in train_ids],
    'annotations': [ann for ann in data['annotations'] if ann['image_id'] in train_ids]
}

val_data = {
    # 'info': data['info'],
    # 'licenses': data['licenses'],
    'categories': data['categories'],
    'images': [img for img in data['images'] if img['id'] in val_ids],
    'annotations': [ann for ann in data['annotations'] if ann['image_id'] in val_ids]
}

with open('train_coco.json', 'w') as f:
    json.dump(train_data, f)

with open('val_coco.json', 'w') as f:
    json.dump(val_data, f)
