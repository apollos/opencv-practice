# USAGE
# python barcode_scanner_image.py --image barcode_example.png

# import the necessary packages
from pyzbar import pyzbar
import zbar
import argparse
import cv2
import numpy as np


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())

# load the input image
image = cv2.imread(args["image"])
hard_limit_high = 1067
hard_limit_width = 680
#resize image
#'''
if image.shape[1] > hard_limit_width:
    image = image_resize(image, width=hard_limit_width)

if image.shape[0] > hard_limit_high:
    image = image_resize(image, height=hard_limit_high)
#'''
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# 1
#blur=cv2.GaussianBlur(image,(0,0),3)
#image=cv2.addWeighted(image,1.5,blur,-0.5,0)
# 2
#kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#image = cv2.filter2D(image, -1, kernel)
# 3
#image=cv2.bilateralFilter(image,9,75,75)
# 4
'''
sigma = 1; threshold = 5; amount = 1
blurred=cv2.GaussianBlur(image,(0,0),1,None,1)
lowContrastMask = abs(image - blurred) < threshold
sharpened = image*(1+amount) + blurred*(-amount)
image=cv2.bitwise_or(sharpened.astype(np.uint8),lowContrastMask.astype(np.uint8))
'''

cv2.imshow("Image", image)
cv2.waitKey(0)
# find the barcodes in the image and decode each of the barcodes
barcodes = pyzbar.decode(image, symbols=[128])

# loop over the detected barcodes
for barcode in barcodes:
    # extract the bounding box location of the barcode and draw the
    # bounding box surrounding the barcode on the image
    (x, y, w, h) = barcode.rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # the barcode data is a bytes object so if we want to draw it on
    # our output image we need to convert it to a string first
    barcodeData = barcode.data.decode("utf-8")
    barcodeType = barcode.type

    # draw the barcode data and barcode type on the image
    text = "{} ({})".format(barcodeData, barcodeType)
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print("Bar Code: {}".format(text))
# show the output image
print(image.shape)
#cv2.imshow("Image", image)
#cv2.waitKey(0)

#image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
for i in range(10):
    config = [('ZBAR_QRCODE', 'ZBAR_CFG_ENABLE', 0), 
                ('ZBAR_CODE39', 'ZBAR_CFG_ENABLE', 0), 
                ('ZBAR_CODE128', 'ZBAR_CFG_ENABLE', 1), 
                ('ZBAR_CODE128', 'ZBAR_CFG_POSITION', 1), 
                ('ZBAR_CODE128', 'ZBAR_CFG_X_DENSITY', i), 
                ('ZBAR_CODE128', 'ZBAR_CFG_Y_DENSITY', i)]
    scanner = zbar.Scanner(config)
    results = scanner.scan(image)
    if results is not None and len(results) > 0:
        for result in results:
            print(result.type, result.data, result.quality, result.position)
        break


