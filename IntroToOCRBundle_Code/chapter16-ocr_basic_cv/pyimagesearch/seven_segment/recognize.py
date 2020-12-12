# import the necessary packages
from .segments import DIGITS_INV
import cv2

def recognize_digit(roi, minArea=0.4):
	# grab the dimensions of the ROI and use those dimensions to
	# define the width and height of each of the 7 segments we are
	# going to examine
	(h, w) = roi.shape[:2]
	(dW, dH, dC) = (int(w * 0.25), int(h * 0.15), int(h * 0.075))

	# define the set of 7 segments
	segments = [
		((0, 0), (w, dH)),  # top
		((0, 0), (dW, h // 2)),  # top-left
		((w - dW, 0), (w, h // 2)),  # top-right
		((0, (h // 2) - dC), (w, (h // 2) + dC)),  # center
		((0, h // 2), (dW, h)),  # bottom-left
		((w - dW, h // 2), (w, h)),  # bottom-right
		((0, h - dH), (w, h))  # bottom
	]

	# initialize an array to store which of the 7 segments are turned
	# on versus not
	on = [0] * len(segments)

	# loop over the segments
	for (i, ((startX, startY), (endX, endY))) in enumerate(segments):
		# extract the segment ROI, count the total number of
		# thresholded pixels in the segment, and then compute
		# the area of the segment
		segROI = roi[startY:endY, startX:endX]
		total = cv2.countNonZero(segROI)
		area = (endX - startX) * (endY - startY)

		# if the total number of non-zero pixels is greater than the
		# minimum percentage of the area, mark the segment as "on"
		if total / float(area) > minArea:
			on[i]= 1

	# OCR the digit using our dictionary
	digit = DIGITS_INV.get(tuple(on), None)

	# return the OCR'd digit
	return digit