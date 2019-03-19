# USAGE
# python detect_barcode.py
# python detect_barcode.py --video video/video_games.mov

# import the necessary packages
from pyimagesearch import simple_barcode_detection
from imutils.video import VideoStream
import argparse
import time
import cv2
import os
import glob

# construct the argument parse and parse the arguments
# change to see result
input_path = "image/rot"
output_path = "image/output"
input_path = os.path.abspath(input_path)
output_path = os.path.abspath(output_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)

file_lst = glob.glob(os.path.join(input_path, "*.jpg"))
for img_file in file_lst:
    # load the image and convert it to grayscale
    image = cv2.imread(img_file)

    # detect the barcode in the image
    box = simple_barcode_detection.detect(image)

    # if a barcode was found, draw a bounding box on the frame
    if box is not None:
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_path, os.path.basename(img_file)), image)


