# USAGE
# python microsoft_ocr.py --image images/aircraft.png

# import the necessary packages
from config import microsoft_cognitive_services as config
import requests
import argparse
import time
import sys
import cv2

def draw_ocr_results(image, text, pts, color=(0, 255, 0)):
	# unpack the points list
	topLeft = pts[0]
	topRight = pts[1]
	bottomRight = pts[2]
	bottomLeft = pts[3]

	# draw the bounding box of the detected text
	cv2.line(image, topLeft, topRight, color, 2)
	cv2.line(image, topRight, bottomRight, color, 2)
	cv2.line(image, bottomRight, bottomLeft, color, 2)
	cv2.line(image, bottomLeft, topLeft, color, 2)

	# draw the text itself
	cv2.putText(image, text, (topLeft[0], topLeft[1] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

	# return the output image
	return image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we'll submit to Microsoft OCR")
args = vars(ap.parse_args())

# load the input image from disk, both in a byte array and OpenCV
# format
imageData = open(args["image"], "rb").read()
image = cv2.imread(args["image"])

# construct our headers dictionary that will include our Microsoft
# Cognitive Services API Key (required in order to submit requests
# to the API)
headers = {
	"Ocp-Apim-Subscription-Key": config.SUBSCRIPTION_KEY,
	"Content-Type": "application/octet-stream",
}

# make the request to the Azure Cognitive Services API and wait for
# a response
print("[INFO] making request to Microsoft Cognitive Services API...")
response = requests.post(config.ENDPOINT_URL, headers=headers,
	data=imageData)
response.raise_for_status()

# initialize whether or not the API request was a success
success = False

# continue to poll the Cognitive Services API for a response until
# either (1) we receive a "success" response or (2) the request fails
while True:
	# check for a response and convert it to a JSON object
	responseFinal = requests.get(
		response.headers["Operation-Location"],
		headers=headers)
	result = responseFinal.json()

	# if the results are available, stop the polling operation
	if "analyzeResult" in result.keys():
		success = True
		break

	# check to see if the request failed
	if "status" in result.keys() and result["status"] == "failed":
		break

	# sleep for a bit before we make another request to the API
	time.sleep(1.0)

# if the request failed, show an error message and exit
if not success:
	print("[INFO] Microsoft Cognitive Services API request failed")
	print("[INFO] Attempting to gracefully exit")
	sys.exit(0)

# grab all OCR'd lines returned by Microsoft's OCR API
lines = result["analyzeResult"]["readResults"][0]["lines"]

# make a copy of the input image for final output
final = image.copy()

# loop over the lines
for line in lines:
	# extract the OCR'd line from Microsoft's API and unpack the
	# bounding box coordinates
	text = line["text"]
	box = line["boundingBox"]
	(tlX, tlY, trX, trY, brX, brY, blX, blY) = box
	pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))

	# draw the output OCR line-by-line
	output = image.copy()
	output = draw_ocr_results(output, text, pts)
	final = draw_ocr_results(final, text, pts)

	# show the output OCR'd line
	print(text)
	cv2.imshow("Output", output)
	cv2.waitKey(0)

# show the final output image
cv2.imshow("Final Output", final)
cv2.waitKey(0)