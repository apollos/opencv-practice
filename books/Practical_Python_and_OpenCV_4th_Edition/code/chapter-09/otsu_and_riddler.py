# USAGE
# python otsu_and_riddler.py --image ../images/coins.png

# Import the necessary packages
from __future__ import print_function
import numpy as np
import argparse
import mahotas
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image")
args = vars(ap.parse_args())

# Load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Image", image)

# OpenCV provides methods to use Otsu's thresholding, but I find
# the mahotas implementation is more 'Pythonic'. Otsu's method
# assumes that are two 'peaks' in the grayscale histogram. It finds
# these peaks, and then returns a value we should threshold on.
T = mahotas.thresholding.otsu(blurred)
print("Otsu's threshold: {}".format(T))

# Applying the threshold can be done using NumPy, where values
# smaller than the threshold are set to zero, and values above
# the threshold are set to 255 (white).
thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Otsu", thresh)

# An alternative is to use the Riddler-Calvard method
T = mahotas.thresholding.rc(blurred)
print("Riddler-Calvard: {}".format(T))
thresh = image.copy()
thresh[thresh > T] = 255
thresh[thresh < 255] = 0
thresh = cv2.bitwise_not(thresh)
cv2.imshow("Riddler-Calvard", thresh)
cv2.waitKey(0)