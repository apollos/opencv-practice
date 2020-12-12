# USAGE
# python localize_text_opencv.py --east ../models/east/frozen_east_text_detection.pb --image images/car_wash.png

# import the necessary packages
from pyimagesearch.east import EAST_OUTPUT_LAYERS
from pyimagesearch.east import decode_predictions
import numpy as np
import argparse
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-e", "--east", required=True,
	help="path to input EAST text detector")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-t", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
ap.add_argument("-c", "--min-conf", type=float, default=0.5,
	help="minimum probability required to inspect a text region")
ap.add_argument("-n", "--nms-thresh", type=float, default=0.4,
	help="non-maximum suppression threshold")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
(origH, origW) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(EAST_OUTPUT_LAYERS)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# decode the predictions form OpenCV's EAST text detector and then
# apply non-maximum suppression (NMS) to the rotated bounding boxes
(rects, confidences) = decode_predictions(scores, geometry,
	minConf=args["min_conf"])
idxs = cv2.dnn.NMSBoxesRotated(rects, confidences,
	args["min_conf"], args["nms_thresh"])

# ensure that at least one text bounding box was found
if len(idxs) > 0:
	# loop over the valid bounding box indexes after applying NMS
	for i in idxs.flatten():
		# compute the four corners of the bounding box, scale the
		# coordinates based on the respective ratios, and then
		# convert the box to an integer NumPy array
		box = cv2.boxPoints(rects[i])
		box[:, 0] *= rW
		box[:, 1] *= rH
		box = np.int0(box)

		# draw a rotated bounding box around the text
		cv2.polylines(image, [box], True, (0, 255, 0), 2)

# show the output image
cv2.imshow("Text Detection", image)
cv2.waitKey(0)