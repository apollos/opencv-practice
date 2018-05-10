# USAGE
# python build_dataset.py

# import the necessary packages
from pyimagesearch.io import HDF5DatasetWriter
from conf import sr_config as config
from imutils import paths
from scipy import misc
import shutil
import random
import cv2
import os

# if the output directories do not exist, create them
for p in [config.IMAGES, config.LABELS]:
	if not os.path.exists(p):
		os.makedirs(p)

# grab the image paths and initialize the total number of crops
# processed
print("[INFO] creating temporary images...")
imagePaths = list(paths.list_images(config.INPUT_IMAGES))
random.shuffle(imagePaths)
total = 0

# loop over the image paths
for imagePath in imagePaths:
	# load the input image
	image = cv2.imread(imagePath)

	# grab the dimensions of the input image and crop the image such
	# that it tiles nicely when we generate the training data +
	# labels
	(h, w) = image.shape[:2]
	w -= int(w % config.SCALE)
	h -= int(h % config.SCALE)
	image = image[0:h, 0:w]

	# to generate our training images we first need to downscale the
	# image by the scale factor...and then upscale it back to the
	# original size -- this will process allows us to generate low
	# resolution inputs that we'll then learn to reconstruct the high
	# resolution versions from
	scaled = misc.imresize(image, 1.0 / config.SCALE,
		interp="bicubic")
	scaled = misc.imresize(scaled, config.SCALE / 1.0,
		interp="bicubic")

	# slide a window from left-to-right and top-to-bottom
	for y in range(0, h - config.INPUT_DIM + 1, config.STRIDE):
		for x in range(0, w - config.INPUT_DIM + 1, config.STRIDE):
			# crop output the `INPUT_DIM x INPUT_DIM` ROI from our
			# scaled image -- this ROI will serve as the input to our
			# network
			crop = scaled[y:y + config.INPUT_DIM,
				x:x + config.INPUT_DIM]

			# crop out the `LABEL_SIZE x LABEL_SIZE` ROI from our
			# original image -- this ROI will be the target output
			# from our network
			target = image[
				y + config.PAD:y + config.PAD + config.LABEL_SIZE,
				x + config.PAD:x + config.PAD + config.LABEL_SIZE]

			# construct the crop and target output image paths
			cropPath = os.path.sep.join([config.IMAGES,
				"{}.png".format(total)])
			targetPath = os.path.sep.join([config.LABELS,
				"{}.png".format(total)])

			# write the images to disk
			cv2.imwrite(cropPath, crop)
			cv2.imwrite(targetPath, target)

			# increment the crop total
			total += 1

# grab the paths to the images
print("[INFO] building HDF5 datasets...")
inputPaths = sorted(list(paths.list_images(config.IMAGES)))
outputPaths = sorted(list(paths.list_images(config.LABELS)))

# initialize the HDF5 datasets
inputWriter = HDF5DatasetWriter((len(inputPaths), config.INPUT_DIM,
	config.INPUT_DIM, 3), config.INPUTS_DB)
outputWriter = HDF5DatasetWriter((len(outputPaths),
	config.LABEL_SIZE, config.LABEL_SIZE, 3), config.OUTPUTS_DB)

# loop over the images
for (inputPath, outputPath) in zip(inputPaths, outputPaths):
	# load the two images and add them to their respective datasets
	inputImage = cv2.imread(inputPath)
	outputImage = cv2.imread(outputPath)
	inputWriter.add([inputImage], [-1])
	outputWriter.add([outputImage], [-1])

# close the HDF5 datasets
inputWriter.close()
outputWriter.close()

# delete the temporary output directories
print("[INFO] cleaning up...")
shutil.rmtree(config.IMAGES)
shutil.rmtree(config.LABELS)