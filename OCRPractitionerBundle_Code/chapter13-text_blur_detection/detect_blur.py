# USAGE
# python detect_blur.py --image images/example_01.png

# import the necessary packages
from pyimagesearch.blur_detection import detect_blur_fft
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path input image that we'll detect blur in")
ap.add_argument("-t", "--thresh", type=int, default=15,
	help="threshold for our blur detector to fire")
args = vars(ap.parse_args())

# load the input image from disk, resize it, and convert it to
# grayscale
orig = cv2.imread(args["image"])
orig = imutils.resize(orig, width=500)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# apply our blur detector using the FFT
(mean, blurry) = detect_blur_fft(gray, size=60,
	thresh=args["thresh"])

# draw on the image, indicating whether or not it is blurry
image = np.dstack([gray] * 3)
color = (0, 0, 255) if blurry else (0, 255, 0)
text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
text = text.format(mean)
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, color, 2)
print("[INFO] {}".format(text))

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)