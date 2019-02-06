# USAGE
# python crop.py --image ../images/trex.png

# Import the necessary packages
import numpy as np
import argparse
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image")
args = vars(ap.parse_args())

# Load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# Cropping an image is as simple as using array slices
# in NumPy! Let's crop out the face of the T-Rex. The
# order in which we specify the coordinates is:
#	startY:endY, startX:endX
# In this case, we are starting at Y=30 and ending at
# Y=120. Similarly, we start at X=240 and X=335.
cropped = image[30:120 , 240:335]
cv2.imshow("T-Rex Face", cropped)
cv2.waitKey(0)