# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np
import copy
import math

def cal_contour_point_val(contours):
    return contours[0][0]+contours[0][1]*800


def find_border_angel(mid_point, image, direction='horizon', reverse_flag=0):
    kernel = [255, 255, 255, 255, 255]
    bias = 100
    image_shape = np.shape(image)
    degree = 0
    if direction == 'horizon':
        bias = min(image_shape[1] - mid_point[0] - int(len(kernel)/2), mid_point[0] - int(len(kernel)/2),
                   bias + int(len(kernel)/2))
        if reverse_flag == 0:
            search_range = range(image_shape[0])
        else:
            search_range = reversed(range(image_shape[0]))
        a_point_y = -1
        b_point_y = -1
        for i in search_range:
            a_tmp = np.dot(image[i, mid_point[0] - bias - 2:mid_point[0] - bias + 3], kernel)
            b_tmp = np.dot(image[i, mid_point[0] + bias - 2:mid_point[0] + bias + 3], kernel)
            if a_tmp >= 130050:
                a_point_y = i
            if b_tmp >= 130050:
                b_point_y = i
            if a_point_y != -1 and b_point_y != -1:
                break
        print("{} - {}".format(b_point_y, a_point_y))
        if a_point_y != -1 and b_point_y != -1:
            degree = math.degrees(math.atan2(b_point_y - a_point_y, 2 * bias))
            mid_point[1] = round(math.tan(math.radians(degree))*bias) + a_point_y

    elif direction == 'vertical':
        bias = min(image_shape[0] - mid_point[1] - int(len(kernel)/2), mid_point[1] - int(len(kernel)/2),
                   bias + int(len(kernel)/2))
        if reverse_flag == 0:
            search_range = range(image_shape[1])
        else:
            search_range = reversed(range(image_shape[1]))
        a_point_x = -1
        b_point_x = -1
        for i in search_range:
            a_tmp = np.dot(image[mid_point[1] - bias - 2:mid_point[1] - bias + 3, i], kernel)
            b_tmp = np.dot(image[mid_point[1] + bias - 2:mid_point[1] + bias + 3, i], kernel)
            if a_tmp >= 130050:
                a_point_x = i
            if b_tmp >= 130050:
                b_point_x = i
            if a_point_x != -1 and b_point_x != -1:
                break
        print("{} - {}".format(b_point_x, a_point_x))
        if a_point_x != -1 and b_point_x != -1:
            degree = 0 - math.degrees(math.atan2(b_point_x - a_point_x, 2 * bias)) #same degree
            mid_point[0] = round(math.tan(math.radians(degree))*bias) + a_point_x
    else:
        print("Error!")
    #print("({}, {})".format(mid_point[0], mid_point[1]))
    return degree

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=800)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
'''
a = np.shape(blurred)
cent_x, cent_y = int(a[1]/2), int(a[0]/2)
for i in range(cent_y):
    print("{}: {}".format(i, blurred[i][cent_x]))
thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)[1]
'''
canny_rst = cv2.Canny(blurred, 50, 150, L2gradient=True)
cv2.imshow("Test", canny_rst)
cv2.waitKey(0)
cnts = cv2.findContours(canny_rst.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
c = max(cnts, key=cv2.contourArea)
c = np.array(np.array(c).astype(float)*ratio).astype(int)
# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# draw the outline of the object, then draw each of the
# extreme points, where the left-most is red, right-most
# is green, top-most is blue, and bottom-most is teal
cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
cv2.circle(image, extLeft, 6, (0, 0, 255), -1)
cv2.circle(image, extRight, 6, (0, 255, 0), -1)
cv2.circle(image, extTop, 6, (255, 0, 0), -1)
cv2.circle(image, extBot, 6, (255, 255, 0), -1)

resize_img = imutils.resize(image, width=800)
# show the output image
cv2.imshow("Image", resize_img)
cv2.waitKey(0)
