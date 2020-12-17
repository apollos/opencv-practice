# USAGE
# python handwritten_to_text.py --name handwriting --image full.png --ground-truth full.txt
# python handwritten_to_text.py --name eng --image full.png --ground-truth full.txt

# import the necessary packages
from difflib import SequenceMatcher as SQ
import pytesseract
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
	help="name of the OCR model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image of handwritten text")
ap.add_argument("-g", "--ground-truth", required=True,
	help="path to text file containing the ground-truth labels")
args = vars(ap.parse_args())

# load the input image and convert it from BGR to RGB channel
# ordering
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# use Tesseract to OCR the image
print("[INFO] OCR'ing the image...")
predictedText = pytesseract.image_to_string(image, lang=args["name"])
print(predictedText)

# read the text from ground truth file
with open(args["ground_truth"], "r") as f:
	target = f.read()

# calculate the accuracy of the model with respect the ratio of
# sequences matched in between the predicted and groud-truth labels
accuracyScore = SQ(None, target, predictedText).ratio() * 100

# round off the accuracy score and print it out
accuracyScore = round(accuracyScore, 2)
print("[INFO] accuracy of {} model: {}%...".format(args["name"],
	accuracyScore))