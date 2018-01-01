# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np
import copy


def cal_contour_point_val(contours):
    return contours[0][0]+contours[0][1]*800

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
# shape detector
contours_tuple = cv2.findContours(canny_rst.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

cnts = contours_tuple[0] if imutils.is_cv2() else contours_tuple[1]
#sd = ShapeDetector()
# loop over the contours
tst_c = cnts[0]
print(tst_c[:, :, 0].argmin())
extLeft = tuple(tst_c[tst_c[:, :, 0].argmin()][0])
print(extLeft)
c = min(tst_c, key=cal_contour_point_val)
for a in tst_c:
    print(a)
print(c)
c = c.astype("float")
c *= ratio
c = c.astype("int")
cv2.circle(image, (c[0][0], c[0][1]+10), 5, (0,0,255), -1)
resized = imutils.resize(image, width=800)
cv2.imshow("Image", resized)
cv2.waitKey(0)
exit(0)
right_top_point_lst = list(map(lambda x: x[0][0]+x[0][1]*800, [min(c, key=cal_contour_point_val) for c in cnts]))
tmp_lst = sorted(right_top_point_lst)
for c in tmp_lst:
    '''
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)
    '''
    plot_c = cnts[right_top_point_lst.index(c)]
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    plot_c = plot_c.astype("float")
    plot_c *= ratio
    plot_c = plot_c.astype("int")
    cv2.drawContours(image, [plot_c], -1, (0, 255, 0), 2)
    '''
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 2)
    '''
    # show the output image
    resized = imutils.resize(image, width=800)
    cv2.imshow("Image", resized)
    cv2.waitKey(0)
cv2.destroyAllWindows()
