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
#laplacian = cv2.Laplacian(blurred,cv2.CV_64F)
canny_rst = cv2.Canny(blurred, 50, 150, L2gradient=True)
cv2.imshow("Test", canny_rst)
cv2.waitKey(0)
# find contours in the thresholded image and initialize the
top_mid = [int(np.shape(canny_rst)[1]/2), 0]
bottom_mid = [int(np.shape(canny_rst)[1]/2), np.shape(canny_rst)[0]]
left_mid = [0, int(np.shape(canny_rst)[0]/2)]
right_mid = [np.shape(canny_rst)[1], int(np.shape(canny_rst)[0]/2)]

'''
kernel = [255, 255, 255, 255, 255]

for i in range(np.shape(canny_rst)[0]):
    if np.dot(canny_rst[i, top_mid[0]-2:top_mid[0]+3], kernel) >= 130050:
        top_mid[1] = i
        break

for i in reversed(range(np.shape(canny_rst)[0])):
    if np.dot(canny_rst[i, bottom_mid[0]-2: bottom_mid[0]+3], kernel) >= 130050:
        bottom_mid[1] = i
        break

for i in range(np.shape(canny_rst)[1]):
    if np.dot(canny_rst[left_mid[1]-2:left_mid[1]+3, i], kernel) >= 130050:
        left_mid[0] = i
        break

for i in reversed(range(np.shape(canny_rst)[1])):
    if np.dot(canny_rst[right_mid[1]-2:right_mid[1]+3, i], kernel) >= 130050:
        right_mid[0] = i
        break

print("({}, {})".format(top_mid[0], top_mid[1]))
print("({}, {})".format(bottom_mid[0], bottom_mid[1]))
print("({}, {})".format(left_mid[0], left_mid[1]))
print("({}, {})".format(right_mid[0], right_mid[1]))
rotate_degree = 0
'''

t_rotate_degree = find_border_angel(top_mid, canny_rst, 'horizon')
b_rotate_degree = find_border_angel(bottom_mid, canny_rst, 'horizon', 1)
l_rotate_degree = find_border_angel(left_mid, canny_rst, 'vertical')
r_rotate_degree = find_border_angel(right_mid, canny_rst, 'vertical', 1)


top_mid = np.array(np.array(top_mid).astype('float') * ratio).astype('int')
bottom_mid = np.array(np.array(bottom_mid).astype('float') * ratio).astype('int')
left_mid = np.array(np.array(left_mid).astype('float') * ratio).astype('int')
right_mid = np.array(np.array(right_mid).astype('float') * ratio).astype('int')
xmin = max(left_mid[0], 0)
ymin = max(top_mid[1], 0)
xmax = min(right_mid[0], np.shape(image)[1])
ymax = min(bottom_mid[1], np.shape(image)[0])

print(t_rotate_degree, b_rotate_degree, l_rotate_degree, r_rotate_degree)
print(xmin, ymin, xmax, ymax)
'''
img_rotation = imutils.rotate(image, (t_rotate_degree - b_rotate_degree + r_rotate_degree - l_rotate_degree)/4)

cv2.rectangle(img_rotation, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
resized = imutils.resize(img_rotation, width=800)
cv2.imshow("Detect", resized)
'''
cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
resized = imutils.resize(image, width=800)
cv2.imshow("No Rotation", resized)

cv2.waitKey(0)

