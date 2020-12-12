# USAGE
# python visualize_digits.py

# import the necessary packages
from pyimagesearch.seven_segment import DIGITS
import numpy as np
import cv2

# initialize the dimensions of our example image and use those
# dimensions to define the width and height of each of the
# 7 segments we are going to examine
(h, w) = (470, 315)
(dW, dH, dC) = (int(w * 0.25), int(h * 0.15), int(h * 0.075))

# define the set of 7 segments
segments = [
	((0, 0), (w, dH)),  # top
	((0, 0), (dW, h // 2)),  # top-left
	((w - dW, 0), (w, h // 2)),	 # top-right
	((0, (h // 2) - dC), (w, (h // 2) + dC)),  # center
	((0, h // 2), (dW, h)),	 # bottom-left
	((w - dW, h // 2), (w, h)),  # bottom-right
	((0, h - dH), (w, h))  # bottom
]

# loop over the digits and associated 7 segment display for that
# particular digit
for (digit, display) in DIGITS.items():
	# allocate memory for the visualization of that digit
	vis = np.zeros((h, w, 3))

	# loop over the segments and whether or not that particular
	# segment is turned on or not
	for (segment, on) in zip(segments, display):
		# verify that the segment is indeed on
		if on:
			# unpack the starting and ending (x, y)-coordinates of
			# the current segment, then draw it on our visualization
			# image
			((startX, startY), (endX, endY)) = segment
			cv2.rectangle(vis, (startX, startY), (endX, endY),
				(0, 0, 255), -1)

	# show the output visualization for the digit
	print("[INFO] visualization for '{}'".format(digit))
	cv2.imshow("Digit", vis)
	cv2.waitKey(0)