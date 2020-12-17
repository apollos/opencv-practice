# USAGE
# python text_detection_video.py --east ../models/east/frozen_east_text_detection.pb
# python text_detection_video.py --east ../models/east/frozen_east_text_detection.pb --use-gpu 1

# import the necessary packages
from pyimagesearch.east import EAST_OUTPUT_LAYERS
from pyimagesearch.east import decode_predictions
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
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

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

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

# if a video path was not supplied, grab the reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["input"])

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame, maintaining the aspect ratio
	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()

	# if our frame dimensions are None, we still need to compute the
	# ratio of old frame dimensions to new frame dimensions
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)

	# construct a blob from the image and then perform a forward pass
	# of the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(EAST_OUTPUT_LAYERS)

	# decode the predictions form OpenCV's EAST text detector and
	# then apply non-maximum suppression (NMS) to the rotated
	# bounding boxes
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
			cv2.polylines(orig, [box], True, (0, 255, 0), 2)

	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Text Detection", orig)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a webcam, release the pointer
if not args.get("input", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()