from mmrcnn import model as modellib, visualize
import os
import sys
import numpy as np
import coco
import skimage.io
from datetime import datetime
import psutil
import cv2

pid = os.getpid()
py = psutil.Process(pid)
#memoryUse = py.get_memory_info()[0]/2.**30
#print("Memory use:", memoryUse)
WEIGHTS_DIR = "./weights"
TEST_PIC_DIR = "./testpictures"
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

config = coco.CocoConfig()
model_path = model_path = os.path.join(WEIGHTS_DIR, "trained_coco_2018-Jun-13__11_02_08.h5")
#model_path = "/home/thiemi/MaskRCNN/Mask_RCNN/mask_rcnn_coco.h5"

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=WEIGHTS_DIR)
#model = modellib.MaskRCNN(mode="inference", config=config, model_dir="/home/thiemi/MaskRCNN/Mask_RCNN")
# returns a compiled model
model.load_weights(model_path, by_name=True)
print("successfully loaded model")

#image = skimage.io.imread(os.path.join(TEST_PIC_DIR, "street" + str(7) + ".jpg"))

image = cv2.imread(os.path.join(TEST_PIC_DIR, "street" + str(7) + ".jpg"))
height, width = image.shape[:2]
if height > width:
    r = 64 / height
    small = cv2.resize(image, (int(width * r)  , 64))
else:
    r = 64 / width
    small = cv2.resize(image, (64, int(height * r)))
if image is None: 
    print("image is null")
#image = skimage.io.imread(os.path.join(TEST_PIC_DIR, "gray.jpg"))
#memoryUse = py.get_memory_info()[0]/2.**30
#print("Memory use:", memoryUse)

# Run detection
start = datetime.now()
result = model.detect([small], verbose=1)
print("Time taken for detection: {}".format(datetime.now() - start))

r = result[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])


