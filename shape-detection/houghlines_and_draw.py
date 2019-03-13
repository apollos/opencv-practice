# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
wide = cv2.Canny(blurred, 10, 80)
horizon = cv2.Canny(blurred, 80, 10)
tight = cv2.Canny(blurred, 225, 250)
auto = imutils.auto_canny(blurred)

# show the images
"""
cv2.imshow("Original", image)
cv2.imshow("Wide", wide)
cv2.imshow("Tight", tight)
cv2.imshow("horizon", horizon)
cv2.imshow("Auto", auto)
cv2.waitKey(0)
"""
im2, contours, hierarchy = cv2.findContours(wide.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
print("{} {} {}".format(np.shape(im2), np.shape(contours), np.shape(hierarchy)))

changed_img = np.zeros(np.shape(im2), dtype=np.uint8)
for idx in range(len(contours)):
    if len(contours[idx]) > 700:
        reshape_contours = np.reshape(contours[idx], (-1, 2))
        for contour in reshape_contours:
            changed_img[contour[1], contour[0]] = 255
cv2.imshow("drawContours", changed_img)
cv2.waitKey(0)

lines = cv2.HoughLines(changed_img, rho=1, theta = np.pi / 180, threshold=70, min_theta=0) #, max_theta=40)
#lines = cv2.HoughLines(changed_img, 1, np.pi / 180, 150, None, 0, 0)
print(len(lines))
painted_lines = []
count = 0
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        same_line = False
        #"""
        for existed_line in painted_lines:
            if rho - 50 <existed_line[0] < rho+50 and theta - 1 < existed_line[1]<theta +1:
                same_line = True
                break
        #"""
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