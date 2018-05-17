from mmrcnn import model as modellib, visualize
import os
import sys
import numpy as np
import coco
import skimage.io
from datetime import datetime

WEIGHTS_DIR = "/net4/merkur/storage/deeplearning/users/thiemi/mmrcnn/weights"
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
model_path = model_path = os.path.join(WEIGHTS_DIR, "trained_coco_2018-May-15__11_08_18.h5")
#model_path = "/home/thiemi/MaskRCNN/Mask_RCNN/mask_rcnn_coco.h5"

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=WEIGHTS_DIR)
#model = modellib.MaskRCNN(mode="inference", config=config, model_dir="/home/thiemi/MaskRCNN/Mask_RCNN")
# returns a compiled model
model.load_weights(model_path, by_name=True)
print("successfully loaded model")
start = datetime.now()
image = skimage.io.imread(os.path.join(TEST_PIC_DIR, "street" + str(7) + ".jpg"))
# Run detection
result = model.detect([image], verbose=1)
print("Time taken for detection: {}".format(datetime.now() - start))

r = result[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])


