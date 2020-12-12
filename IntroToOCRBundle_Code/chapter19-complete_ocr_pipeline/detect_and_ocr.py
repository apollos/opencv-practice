# USAGE
# python detect_and_ocr.py --east ../models/east/frozen_east_text_detection.pb --image images/car_wash.png

# import the necessary packages
from pyimagesearch.east import EAST_OUTPUT_LAYERS
from pyimagesearch.east import decode_predictions
from pyimagesearch.helpers import cleanup_text
import pytesseract
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
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
ap.add_argument("-s", "--sort", type=str, default="top-to-bottom",
	help="whether we sort bounding boxes left-to-right or top-to-bottom")
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

# initialize our list of results
results = []

# loop over the valid bounding box indexes after applying NMS
for i in idxs.flatten():
	# compute the four corners of the bounding box, scale the
	# coordinates based on the respective ratios, and then
	# convert the box to an integer NumPy array
	box = cv2.boxPoints(rects[i])
	box[:, 0] *= rW
	box[:, 1] *= rH
	box = np.int0(box)

	# we can only OCR *normal* bounding boxes (i.e., non-rotated
	# ones), so we must convert the bounding box
	(x, y, w, h) = cv2.boundingRect(box)

	# in order to obtain a better OCR of the text we can potentially
	# apply a bit of padding surrounding the bounding box -- here we
	# are computing the deltas in both the x and y directions
	dX = int(w * args["padding"])
	dY = int(h * args["padding"])

	# apply padding to each side of the bounding box
	startX = max(0, x - dX)
	startY = max(0, y - dY)
	endX = min(origW, x + w + (dX * 2))
	endY = min(origH, y + h + (dY * 2))

	# extract the padded ROI
	paddedROI = image[startY:endY, startX:endX]

	# use Tesseract to OCR the ROI
	options = "--psm 7"
	text = pytesseract.image_to_string(paddedROI, config=options)

	# add the rotated bounding box and OCR'd text to our results list
	results.append((box, text))

# check to see if we should sort the bounding boxes (and associated
# OCR'd text) from top-to-bottom
if args["sort"] == "top-to-bottom":
	results = sorted(results, key=lambda y: y[0][0][1])

# otherwise, we'll sort them left-to-right
else:
	results = sorted(results, key=lambda x: x[0][0][0])

# loop over the results
for (box, text) in results:
	# display the text OCR'd by Tesseract
	print("{}\n".format(text))

	# draw a rotated bounding box around the text
	output = image.copy()
	cv2.polylines(output, [box], True, (0, 255, 0), 2)

	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV, then draw the text on the output image
	text = cleanup_text(text)
	(x, y, w, h) = cv2.boundingRect(box)
	cv2.putText(output, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
		1.2, (0, 0, 255), 3)

	# show the output image
	cv2.imshow("Text Detection", output)
	cv2.waitKey(0)