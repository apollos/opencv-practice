# USAGE
# python ocr_template_match.py --image images/credit_card_01.png

# import the necessary packages
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import sys
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-r", "--reference", type=str,
	default="ocr_a_reference.png",
	help="path to reference OCR-A image")
args = vars(ap.parse_args())

# define a dictionary that maps the first digit of a credit card
# number to the credit card type
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}

# load the OCR-A reference image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as white on a black
# background
ref = cv2.imread(args["reference"])
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# find contours in the OCR-A image and sort them from left-to-right
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = sort_contours(refCnts, method="left-to-right")[0]

# initialize a dictionary to map digit name to the corresponding
# reference ROI
digits = {}

# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	# update the digits dictionary, mapping the digit name to the ROI
	digits[i] = roi

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=400)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)

# initialize a rectangular (wider than it is tall) to isolate the
# credit card number from the rest of the image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 7))

# apply a tophat (whitehat) morphological operator to find light
# regions against a dark background (i.e., the credit card numbers)
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("Tophat", tophat)

# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
grad = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
grad = np.absolute(grad)
(minVal, maxVal) = (np.min(grad), np.max(grad))
grad = (grad - minVal) / (maxVal - minVal)
grad = (grad * 255).astype("uint8")
cv2.imshow("Gradient", grad)

# apply a closing operation using the rectangular kernel to help
# close gaps in between credit card number digits, then apply
# Otsu's thresholding method to binarize the image
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
thresh = cv2.threshold(grad, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Rect Close", thresh)

# find contours in the image and sort them from top to bottom
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="top-to-bottom")[0]

# initialize the contours that correspond to the digit groups
locs = None

# loop over the contours
for i in range(0, len(cnts) - 4):
	# grab the subset of contours we are looking at and initialize
	# the total difference in y-coordinate values
	subset = cnts[i:i + 4]
	yDiff = 0

	# loop over the subset of contours
	for j in range(0, len(subset) - 1):
		# compute the bounding box coordinates of each contour
		(xA, yA, wA, hA) = cv2.boundingRect(subset[j])
		(xB, yB, wB, hB) = cv2.boundingRect(subset[j + 1])

		# compute the absolute difference between the y-coordinate of
		# the bounding boxes and add it to to the y-coordinate
		# difference accumulator
		yDiff += np.abs(yA - yB)

	# if there is less than a 5 pixel difference between the
	# y-coordinates, then we know we have found our digit groups
	if yDiff < 5:
		locs = [cv2.boundingRect(c) for c in subset]
		break

# if the group locations are None, then we could not find the digits
# in the credit card image, so exit the script
if locs is None:
	print("[INFO] digit groups could not be found")
	sys.exit(0)

# sort the digit locations from left-to-right, then initialize the
# list of classified digits
locs = sorted(locs, key=lambda x:x[0])
output = []

# loop over the 4 groupings of 4 digits
for (gX, gY, gW, gH) in locs:
	# initialize the list of group digits
	groupOutput = []

	# extract the group ROI of 4 digits from the grayscale image and
	# apply thresholding to segment the digits from the background of
	# the credit card
	group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
	group = cv2.threshold(group, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# detect the contours of each individual digit in the group, then
	# sort the digit contours from left to right
	digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	digitCnts = imutils.grab_contours(digitCnts)
	digitCnts = sort_contours(digitCnts, method="left-to-right")[0]

	# loop over the digit contours
	for c in digitCnts:
		# compute the bounding box of the individual digit, extract
		# the digit, and resize it to have the same fixed size as
		# the reference OCR-A images
		(x, y, w, h) = cv2.boundingRect(c)
		roi = group[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))

		# initialize a list of template matching scores
		scores = []

		# loop over the reference digit name and digit ROI
		for (digit, digitROI) in digits.items():
			# apply correlation-based template matching, take the
			# score, and update the scores list
			result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)

		# the classification for the digit ROI will be the reference
		# digit name with the *largest* template matching score
		groupOutput.append(str(np.argmax(scores)))

	# draw the digit classifications around the group
	cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5),
		(0, 0, 255), 2)
	cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

	# update the output digits list
	output.extend(groupOutput)

# determine the card type
cardType = FIRST_NUMBER.get(output[0], "Unknown")

# display the output credit card information to the screen
print("Credit Card Type: {}".format(cardType))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
