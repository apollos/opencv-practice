import cv2

img = cv2.imread(r'images/image_2.tif')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#laplacian = cv2.Laplacian(gray,cv2.CV_64F)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
ret,bw = cv2.threshold(blurred,210,255,cv2.THRESH_BINARY_INV)
cv2.imshow('Win', bw)
cv2.waitKey(0)
_,contours,hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
cntLen = 50
ct = 0 #number of contours
for cnt in contours:
    if len(cnt) > cntLen: #eliminate the noises
        ct += 1
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if w / float(h) >= 0.8 and w / float(h) < 1.1 and area < 720 and area > 400:
            #newimg = img.copy()
            newimg = img
            cv2.drawContours(newimg,[cnt],0,(0,0,255),2)
            print(w / float(h), area)
            cv2.imshow('Win', newimg)
            cv2.waitKey(0)
print('Total contours: ',ct)