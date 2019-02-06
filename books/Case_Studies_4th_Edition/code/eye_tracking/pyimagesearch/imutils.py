# Import the necessary packages
import numpy as np
import cv2

def grab_contours(cnts):
	# if the length the contours tuple returned by cv2.findContours
	# is '2' then we are using either OpenCV v2.4, v4-beta, or
	# v4-official
	if len(cnts) == 2:
		cnts = cnts[0]

	# if the length of the contours tuple is '3' then we are using
	# either OpenCV v3, v4-pre, or v4-alpha
	elif len(cnts) == 3:
		cnts = cnts[1]

	# otherwise OpenCV has changed their cv2.findContours return
	# signature yet again and I have no idea WTH is going on
	else:
		raise Exception(("Contours tuple must have length 2 or 3, "
			"otherwise OpenCV changed their cv2.findContours return "
			"signature yet again. Refer to OpenCV's documentation "
			"in that case"))

	# return the actual contours array
	return cnts

def translate(image, x, y):
	# Define the translation matrix and perform the translation
	M = np.float32([[1, 0, x], [0, 1, y]])
	shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

	# Return the translated image
	return shifted

def rotate(image, angle, center = None, scale = 1.0):
	# Grab the dimensions of the image
	(h, w) = image.shape[:2]

	# If the center is None, initialize it as the center of
	# the image
	if center is None:
		center = (w / 2, h / 2)

	# Perform the rotation
	M = cv2.getRotationMatrix2D(center, angle, scale)
	rotated = cv2.warpAffine(image, M, (w, h))

	# Return the rotated image
	return rotated

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized