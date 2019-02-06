# USAGE
# python simple_thresholding.py --image ../images/coins.png

# Import the necessary packages
import numpy as np
import argparse
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

# Let's apply basic thresholding. The first parameter is the
# image we want to threshold, the second value is is our threshold
# check. If a pixel value is greater than our threshold (in this
# case, 155), we it to be WHITE, otherwise it is BLACK.
(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Binary Inverse", threshInv)

# Using a normal we can change the last argument in the function
# to make the coins black rather than white.
(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)

# Finally, let's use our threshold as a mask and visualize only
# the coins in the image
cv2.imshow("Coins", cv2.bitwise_and(image, image, mask = threshInv))
cv2.waitKey(0)