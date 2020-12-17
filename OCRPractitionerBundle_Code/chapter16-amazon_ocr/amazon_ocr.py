# USAGE
# python amazon_ocr.py --image images/aircraft.png
# python amazon_ocr.py --image images/aircraft.png --type word

# import the necessary packages
from config import aws_config as config
import argparse
import boto3
import cv2

def draw_ocr_results(image, text, poly, color=(0, 255, 0)):
	# unpack the bounding box, taking care to scale the coordinates
	# relative to the input image size
	(h, w) = image.shape[:2]
	tlX = int(poly[0]["X"] * w)
	tlY = int(poly[0]["Y"] * h)
	trX = int(poly[1]["X"] * w)
	trY = int(poly[1]["Y"] * h)
	brX = int(poly[2]["X"] * w)
	brY = int(poly[2]["Y"] * h)
	blX = int(poly[3]["X"] * w)
	blY = int(poly[3]["Y"] * h)

	# build a list of points and use it to construct each vertex
	# of the bounding box
	pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))
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
	help="path to input image that we'll submit to AWS Rekognition")
ap.add_argument("-t", "--type", type=str, default="line",
	choices=["line", "word"],
	help="output text type (either 'line' or 'word')")
args = vars(ap.parse_args())

# connect to AWS so we can use the Rekognition OCR API
client = boto3.client(
	"rekognition",
	aws_access_key_id=config.ACCESS_KEY,
	aws_secret_access_key=config.SECRET_KEY,
	region_name=config.REGION)

# load the input image as a raw binary file and make a request to
# the Rekognition OCR API
print("[INFO] making request to AWS Rekognition API...")
image = open(args["image"], "rb").read()
response = client.detect_text(Image={"Bytes": image})

# grab the text detection results from the API and load the input
# image again, this time in OpenCV format
detections = response["TextDetections"]
image = cv2.imread(args["image"])

# make a copy of the input image for final output
final = image.copy()

# loop over the text detection bounding boxes
for detection in detections:
	# extract the OCR'd text, text type, and bounding box coordinates
	text = detection["DetectedText"]
	textType = detection["Type"]
	poly = detection["Geometry"]["Polygon"]

	# only draw show the output of the OCR process if we are looking
	# at the correct text type
	if args["type"] == textType.lower():
		# draw the output OCR line-by-line
		output = image.copy()
		output = draw_ocr_results(output, text, poly)
		final = draw_ocr_results(final, text, poly)

		# show the output OCR'd line
		print(text)
		cv2.imshow("Output", output)
		cv2.waitKey(0)

# show the final output image
cv2.imshow("Final Output", final)
cv2.waitKey(0)