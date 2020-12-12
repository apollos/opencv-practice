# USAGE
# python ocr_7segment_display.py --image alarm_clock.png

# import the necessary packages
from pyimagesearch.seven_segment import recognize_digit
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input 7-segment display image")
args = vars(ap.parse_args())

# load the input image from disk
image = cv2.imread(args["image"])

# pre-process the image by resizing it, converting it to grayscale,
# blurring it, and thresholding it
image = imutils.resize(image, width=400)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# find contours in the edge map, then sort them by their size in
# descending order
cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# initialize the list of OCR'd digits
digits = []

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour, and then determine if
	# the bounding box passes our width and height tests
	(x, y, w, h) = cv2.boundingRect(c)
	passWidth = (w >= 50 and w <= 70)
	passHeight = (h >= 95 and h <= 115)

	# verify that the contour passes both tests
	if passWidth and passHeight:
		# extract the ROI of the digit and then recognize it
		roi = thresh[y:y + h, x:x + w]
		digit = recognize_digit(roi)

		# verify that our digit was OCR'd
		if digit is not None:
			# update our list of digits and draw the digit on the
			# image
			digits.append(digit)
			cv2.rectangle(image, (x, y), (x + w, y + h),
				(0, 255, 0), 2)
			cv2.putText(image, str(digit), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# display the time to our screen
formattedTime = "{}:{}{}" if len(digits) == 3 else "{}{}:{}{}"
formattedTime = formattedTime.format(*digits)
print("[INFO] OCR'd time: {}".format(formattedTime))

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)