# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np


def calculate_slop(start_contour, end_contour):
    x_delta = abs(end_contour[0] - start_contour[0])
    y_delta = abs(end_contour[1] - start_contour[1])
    return np.inf if y_delta == 0 else float(x_delta) / float(y_delta)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=640)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#thresh = cv2.threshold(blurred, 135, 255, cv2.THRESH_TOZERO)[1]
thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,7,2)
cv2.imshow("Image", thresh)
cv2.waitKey(0)
# find contours in the thresholded image and initialize the
# shape detector
im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
print("{} {} {}".format(np.shape(im2), np.shape(contours), np.shape(hierarchy)))
cnts = imutils.grab_contours((im2, contours, hierarchy))
#print("{} {} {}".format(np.shape(cnts[0]), np.shape(cnts[1]), np.shape(cnts[2])))
"""
for idx in range(len(contours)):
    print(np.shape(contours[idx]))
"""


cv2.drawContours(resized, contours, -1, (255,255,0), 3)
cv2.imshow("drawContours", resized)
cv2.waitKey(0)



changed_img = np.zeros(np.shape(im2), dtype=np.uint8)
for idx in range(len(contours)):
    if len(contours[idx]) > 50:
        reshape_contours = np.reshape(contours[idx], (-1, 2))
        print("Hierarchy: {}".format(hierarchy[0][idx]))
        for contour in reshape_contours:
            changed_img[contour[1], contour[0]] = 255
cv2.imshow("drawContours", changed_img)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
changed_img = cv2.morphologyEx(changed_img, cv2.MORPH_CLOSE, kernel)
cv2.imshow("closed", changed_img)
cv2.waitKey(0)

lines = cv2.HoughLines(changed_img, rho=1, theta = np.pi / 180, threshold=60, min_theta=0) #, max_theta=40) np.pi / 180
#lines = cv2.HoughLines(changed_img, 1, np.pi / 180, 150, None, 0, 0)
print(len(lines))
painted_lines = []
count = 0
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        same_line = False
        for existed_line in painted_lines:
            if rho - 30 <existed_line[0] < rho+30 and theta - 2 < existed_line[1]<theta +2:
                same_line = True
                break
        if same_line:
            painted_lines.append([rho, theta])
            continue
        painted_lines.append([rho, theta])
        count += 1
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(image, pt1, pt2, (0, 0, 255), 3)

cv2.imshow("lines", image)
cv2.waitKey(0)
print(count)
print(painted_lines)
'''
lines = []
start_point = reshape_contours[0]
end_point = reshape_contours[1]

slop = calculate_slop(start_point, end_point)
for contour in reshape_contours[2:]:
    new_slop = calculate_slop(start_point, contour)
    if new_slop == slop:
        # still same line, continue
    """
    print(contour)
    resized[contour[1], contour[0]] = (255, 255, 255)
    cv2.imshow("Image", resized)
    cv2.waitKey(0)
    """

'''
