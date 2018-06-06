"""
Mobile Mask R-CNN Train & Eval Script
for Training on the COCO Dataset

written by github.com/GustavZ

to use tensorboard run inside model_dir with file "events.out.tfevents.123":
tensorboard --logdir="$(pwd)"
"""

## Import Packages
import os
import sys
import imgaug
import time

## Import Mobile Mask R-CNN
from mmrcnn import model as modellib, utils
import coco

## Paths
ROOT_DIR = os.getcwd()
#MODEL_DIR = "/net4/merkur/storage/deeplearning/users/thiemi/mmrcnn/models"
MODEL_DIR = "./weights"
COCO_DIR = "/net4/merkur/storage/deeplearning/users/thiemi/coco"
WEIGHTS_DIR = "/net4/merkur/storage/deeplearning/users/thiemi/mmrcnn/weights"
DEFAULT_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "mask_rcnn_256_cocoperson_0283.h5")

## Dataset
class_names = None #['person']  # all classes: None
dataset_train = coco.CocoDataset()
dataset_train.load_coco(COCO_DIR, "train", class_names=class_names)
dataset_train.prepare()
dataset_val = coco.CocoDataset()
dataset_val.load_coco(COCO_DIR, "val", class_names=class_names)
dataset_val.prepare()

## Model
config = coco.CocoConfig()
config.display()
model = modellib.MaskRCNN(mode="training", model_dir = MODEL_DIR, config=config)
model.keras_model.summary()

## Weights
#model_path = model.get_imagenet_weights()
#model_path = model.find_last()[1]
#model_path = DEFAULT_MODEL_DIR
#print("> Loading weights from {}".format(model_path))
#model.load_weights(model_path, by_name=True)

## Training - Config
starting_epoch = model.epoch
epoch = dataset_train.dataset_size // (config.STEPS_PER_EPOCH * config.BATCH_SIZE)
epochs_warmup = 1* epoch
#epochs_heads = 7 * epoch #+ starting_epoch
epochs_heads = 2 * epoch + starting_epoch
epochs_stage4 = 2 * epoch + starting_epoch
epochs_all = 2 * epoch + starting_epoch
#epochs_stage4 = 7 * epoch #+ starting_epoch
#epochs_all = 7 * epoch #+ starting_epoch
epochs_breakOfDawn = 5 * epoch
augmentation = imgaug.augmenters.Fliplr(0.5)
print("> Training Schedule: \
    \nwarmup: {} epochs \
    \nheads: {} epochs \
    \nstage4+: {} epochs \
    \nall layers: {} epochs \
    \ntill the break of Dawn: {} epochs".format(
    epochs_warmup,epochs_heads,epochs_stage4,epochs_all,epochs_breakOfDawn))

## Training - WarmUp
#print("> Warmup all layers")
#model.train(dataset_train, dataset_val,
#            learning_rate=config.LEARNING_RATE / 10,
#            epochs=epochs_warmup,
#            layers='all',
#            augmentation=augmentation)

## Training - WarmUp Stage
#print("> Warm Up all layers")
#model.train(dataset_train, dataset_val,
#            learning_rate=config.LEARNING_RATE / 10,
#            epochs=epochs_warmup,
#            layers='all',
#            augmentation=augmentation)

## Training - Stage 1
print("> Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epochs_warmup + epochs_heads,
            layers='heads',
            augmentation=augmentation)

## Training - Stage 2
# Finetune layers  stage 4 and up
print("> Fine tune {} stage 4 and up".format(config.BACKBONE))
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epochs_warmup + epochs_heads + epochs_stage4,
            layers="4+",
            augmentation=augmentation)

## Training - Stage 3
# Fine tune all layers
print("> Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=epochs_warmup + epochs_heads + epochs_stage4 + epochs_all,
            layers='all',
            augmentation=augmentation)


moment=time.strftime("%Y-%b-%d__%H_%M_%S",time.localtime())
model.save_weights("{}/trained_coco_{}.h5".format(MODEL_DIR, moment))
#model.save_weights("/net4/merkur/storage/deeplearning/users/thiemi/mmrcnn/weights/new_trained_coco.h5")

## Training - Stage 3
# Fine tune all layers
print("> Fine tune all layers till the break of Dawn")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 100,
            epochs=epochs_warmup + epochs_heads + epochs_stage4 + epochs_all + epochs_breakOfDawn,
            layers='all',
            augmentation=augmentation)

