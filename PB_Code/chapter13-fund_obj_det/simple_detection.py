# USAGE
# python simple_detection.py --image beagle.png --confidence 0.75

# import the necessary packages
from pyimagesearch.utils.simple_obj_det import image_pyramid
from pyimagesearch.utils.simple_obj_det import sliding_window
from pyimagesearch.utils.simple_obj_det import classify_batch
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize variables used for the object detection procedure
INPUT_SIZE = (350, 350)
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (224, 224)
BATCH_SIZE = 64

# load our the network weights from disk
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=True)

# initialize the object detection dictionary which maps class labels
# to their predicted bounding boxes and associated probability
labels = {}

# load the input image from disk and grab its dimensions
orig = cv2.imread(args["image"])
(h, w) = orig.shape[:2]

# resize the input image to be a square
resized = cv2.resize(orig, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

# initialize the batch ROIs and (x, y)-coordinates
batchROIs = None
batchLocs = []

# start the timer
print("[INFO] detecting objects...")
start = time.time()

# loop over the image pyramid
for image in image_pyramid(resized, scale=PYR_SCALE,
	minSize=ROI_SIZE):
	# loop over the sliding window locations
	for (x, y, roi) in sliding_window(image, WIN_STEP, ROI_SIZE):
		# take the ROI and pre-process it so we can later classify the
		# region with Keras
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)
		roi = imagenet_utils.preprocess_input(roi)

		# if the batch is None, initialize it
		if batchROIs is None:
			batchROIs = roi

		# otherwise, add the ROI to the bottom of the batch
		else:
			batchROIs = np.vstack([batchROIs, roi])

		# add the (x, y)-coordinates of the sliding window to the batch
		batchLocs.append((x, y))

		# check to see if our batch is full
		if len(batchROIs) == BATCH_SIZE:
			# classify the batch, then reset the batch ROIs and
			# (x, y)-coordinates
			labels = classify_batch(model, batchROIs, batchLocs,
				labels, minProb=args["confidence"])

			# reset the batch ROIs and (x, y)-coordinates
			batchROIs = None
			batchLocs = []

# check to see if there are any remaining ROIs that still need to be
# classified
if batchROIs is not None:
	labels = classify_batch(model, batchROIs, batchLocs, labels,
		minProb=args["confidence"])

# show how long the detection process took
end = time.time()
print("[INFO] detections took {:.4f} seconds".format(end - start))

# loop over the labels for each of detected objects in the image
for k in labels.keys():
	# clone the input image so we can draw on it
	clone = resized.copy()

	# loop over all bounding boxes for the label and draw them on
	# the image
	for (box, prob) in labels[k]:
		(xA, yA, xB, yB) = box
		cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show the image *without* apply non-maxima suppression
	cv2.imshow("Without NMS", clone)
	clone = resized.copy()

	# grab the bounding boxes and associated probabilities for each
	# detection, then apply non-maxima suppression to suppress
	# weaker, overlapping detections
	boxes = np.array([p[0] for p in labels[k]])
	proba = np.array([p[1] for p in labels[k]])
	boxes = non_max_suppression(boxes, proba)

	# loop over the bounding boxes again, this time only drawing the
	# ones that were *not* suppressed
	for (xA, yA, xB, yB) in boxes:
		cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 0, 255), 2)

	# show the output image
	print("[INFO] {}: {}".format(k, len(boxes)))
	cv2.imshow("With NMS", clone)
	cv2.waitKey(0)