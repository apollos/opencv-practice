# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=1024)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

thresh = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Image", thresh)
cv2.waitKey(0)
# find contours in the thresholded image and initialize the
# shape detector
im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
print("{} {} {}".format(np.shape(im2), np.shape(contours), np.shape(hierarchy)))
cnts = imutils.grab_contours((im2, contours, hierarchy))
#print("{} {} {}".format(np.shape(cnts[0]), np.shape(cnts[1]), np.shape(cnts[2])))
cv2.imshow("Image", im2)
cv2.waitKey(0)
cv2.drawContours(resized, contours, -1, (255,255,0), 3)
cv2.imshow("Image", resized)
cv2.waitKey(0)
