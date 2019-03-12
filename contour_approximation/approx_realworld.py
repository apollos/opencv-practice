# USAGE
# python approx_realworld.py

# import the necessary images
import cv2
import numpy as np
import argparse
import imutils

# load the receipt image and convert, it to grayscale, and detect
# edges
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())



# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#edged = cv2.Canny(gray, 75, 200)
edged = cv2.Canny(gray, 30, 150)
'''
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = imutils.auto_canny(blurred)
'''
# show the original image and edged map
cv2.imshow("Original", image)
cv2.imshow("Edge Map", edged)


# find contours in the image and sort them from largest to smallest,
# keeping only the largest ones
values = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("{} {} {}".format(np.shape(values[0]), np.shape(values[1]), np.shape(values[2])))
cnts = values[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:7]

# loop over the contours
for c in cnts:
	# approximate the contour and initialize the contour color
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.01 * peri, True)

	# show the difference in number of vertices between the original
	# and approximated contours
	print ("original: {}, approx: {}".format(len(c), len(approx)))

	# if the approximated contour has 4 vertices, then we have found
	# our rectangle
	if len(approx) == 4:
		# draw the outline on the image
		cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)