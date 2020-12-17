# USAGE
# python text_detection_speed.py --image images/car_wash.png --east ../models/east/frozen_east_text_detection.pb
# python text_detection_speed.py --image images/car_wash.png --east ../models/east/frozen_east_text_detection.pb --use-gpu 1

# import the necessary packages
from pyimagesearch.east import EAST_OUTPUT_LAYERS
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
ap.add_argument("-g", "--use-gpu", type=bool, default=False,
	help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# check if we are going to use GPU
if args["use_gpu"]:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# otherwise we are using our CPU
else:
	print("[INFO] using CPU for inference...")

# load the input image and then set the new width and height values
# based on our command line arguments
image = cv2.imread(args["image"])
(newW, newH) = (args["width"], args["height"])

# construct a blob from the image, set the blob as input to the
# network, and initialize a list that records the amount of time
# each forward pass takes
print("[INFO] running timing trials...")
blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
timings = []

# loop over 500 trials to obtain a good approximation to how long
# each forward pass will take
for i in range(0, 500):
	# time the forward pass
	start = time.time()
	(scores, geometry) = net.forward(EAST_OUTPUT_LAYERS)
	end = time.time()
	timings.append(end - start)

# show average timing information on text prediction
avg = np.mean(timings)
print("[INFO] avg. text detection took {:.6f} seconds".format(avg))