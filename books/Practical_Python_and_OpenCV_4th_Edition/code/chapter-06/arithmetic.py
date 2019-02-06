# USAGE
# python arithmetic.py --image ../images/trex.png

# Import the necessary packages
from __future__ import print_function
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

# Images are NumPy arrays, stored as unsigned 8 bit integers.
# What does this mean? It means that the values of our pixels
# will be in the range [0, 255]. When using functions like
# cv2.add and cv2.subtract, values will be clipped to this
# range, even if the added or subtracted values fall outside
# the range of [0, 255]. Check out an example:
print("max of 255: {}".format(cv2.add(np.uint8([200]), np.uint8([100]))))
print("min of 0: {}".format(cv2.subtract(np.uint8([50]), np.uint8([100]))))

# NOTE: If you use NumPy arithmetic operations on these arrays,
# the values will be modulos (wrap around) instead of being
# clipped to the [0, 255] arrange. This is important to keep
# in mind when working with images.
print("wrap around: {}".format(np.uint8([200]) + np.uint8([100])))
print("wrap around: {}".format(np.uint8([50]) - np.uint8([100])))

# Let's increase the intensity of all pixels in our image
# by 100. We accomplish this by constructing a NumPy array
# that is the same size of our matrix (filled with ones)
# and the multiplying it by 100 to create an array filled
# with 100's. Then we simply add the images together. Notice
# how the image is "brighter".
M = np.ones(image.shape, dtype = "uint8") * 100
added = cv2.add(image, M)
cv2.imshow("Added", added)

# Similarly, we can subtract 50 from all pixels in our
# image and make it darker:
M = np.ones(image.shape, dtype = "uint8") * 50
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted", subtracted)
cv2.waitKey(0)