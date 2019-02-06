# USAGE
# python adaptive_thresholding.py --image ../images/coins.png

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

# In our previous example, we had to use manually specify a
# pixel value to globally threshold the image. In this example
# we are going to examine a neighborbood of pixels and adaptively
# apply thresholding to each neighborbood. In this example, we'll
# calculate the mean value of the neighborhood area of 11 pixels
# and threshold based on that value. Finally, our constant C is
# subtracted from the mean calculation (in this case 4)
thresh = cv2.adaptiveThreshold(blurred, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
cv2.imshow("Mean Thresh", thresh)

# We can also apply Gaussian thresholding in the same manner
thresh = cv2.adaptiveThreshold(blurred, 255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
cv2.imshow("Gaussian Thresh", thresh)
cv2.waitKey(0)