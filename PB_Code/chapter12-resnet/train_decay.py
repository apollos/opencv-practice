# USAGE
# python train_decay.py --model output/resnet_tinyimagenet_decay.hdf5 --output output

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import ResNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
import argparse
import json
import sys
import os

# set a high recursion limit so Theano doesn't complain
sys.setrecursionlimit(5000)

# define the total number of epochs to train for along with the
# initial learning rate
NUM_EPOCHS = 75
INIT_LR = 1e-1

def poly_decay(epoch):
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0

	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	# return the new learning rate
	return alpha

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 64, aug=aug,
	preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 64,
	preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

# TODO:
# Change `figPath` and `jsonPath` to correctly use the `FIG_PATH`
# and `JSON_PATH` in the `tiny_imagenet_config`?

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(
	os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(
	os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath),
	LearningRateScheduler(poly_decay)]

# initialize the optimizer and model (ResNet-56)
print("[INFO] compiling model...")
model = ResNet.build(64, 64, 3, config.NUM_CLASSES, (3, 4, 6),
	(64, 128, 256, 512), reg=0.0005, dataset="tiny_imagenet")
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // 64,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // 64,
	epochs=NUM_EPOCHS,
	max_q_size=64 * 2,
	callbacks=callbacks, verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save(args["model"])

# close the databases
trainGen.close()
valGen.close()