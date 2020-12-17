# USAGE
# python denoise_document.py --testing denoising-dirty-documents/test

# import the necessary packages
from config import denoise_config as config
from pyimagesearch.denoising import blur_and_threshold
from imutils import paths
import argparse
import pickle
import random
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--testing", required=True,
	help="path to directory of testing images")
ap.add_argument("-s", "--sample", type=int, default=10,
	help="sample size for testing images")
args = vars(ap.parse_args())

# load our document denoiser from disk
model = pickle.loads(open(config.MODEL_PATH, "rb").read())

# grab the paths to all images in the testing directory and then
# randomly sample them
imagePaths = list(paths.list_images(args["testing"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:args["sample"]]

# loop over the sampled image paths
for imagePath in imagePaths:
	# load the image, convert it to grayscale, and clone it
	print("[INFO] processing {}".format(imagePath))
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	orig = image.copy()

	# pad the image followed by blurring/thresholding it
	image = cv2.copyMakeBorder(image, 2, 2, 2, 2,
		cv2.BORDER_REPLICATE)
	image = blur_and_threshold(image)

	# initialize a list to store our ROI features (i.e., 5x5 pixel
	# neighborhoods)
	roiFeatures = []

	# slide a 5x5 window across the image
	for y in range(0, image.shape[0]):
		for x in range(0, image.shape[1]):
			# extract the window ROI and grab the spatial dimensions
			roi = image[y:y + 5, x:x + 5]
			(rH, rW) = roi.shape[:2]

			# if the ROI is not 5x5, throw it out
			if rW != 5 or rH != 5:
				continue

			# our features will be the flattened 5x5=25 pixels from
			# the training ROI
			features = roi.flatten()
			roiFeatures.append(features)

	# use the ROI features to predict the pixels of our new denoised
	# image
	pixels = model.predict(roiFeatures)

	# the pixels list is currently a 1D array so we need to reshape
	# it to a 2D array (based on the original input image dimensions)
	# and then scale the pixels from the range [0, 1] to [0, 255]
	pixels = pixels.reshape(orig.shape)
	output = (pixels * 255).astype("uint8")

	# show the original and output images
	cv2.imshow("Original", orig)
	cv2.imshow("Output", output)
	cv2.waitKey(0)