# USAGE
# python detect_smile.py --cascade haarcascade_frontalface_default.xml \
#	--model output/lenet.hdf5 

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame, then we
	# have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, convert it to grayscale, and then clone the
	# original frame so we can draw on it later in the program
	frame = imutils.resize(frame, width=300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frameClone = frame.copy()

	# detect faces in the input frame, then clone the frame so that
	# we can draw on it
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# loop over the face bounding boxes
	for (fX, fY, fW, fH) in rects:
		# extract the ROI of the face from the grayscale image,
		# resize it to a fixed 28x28 pixels, and then prepare the
		# ROI for classification via the CNN
		roi = gray[fY:fY + fH, fX:fX + fW]
		roi = cv2.resize(roi, (28, 28))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)

		# determine the probabilities of both "smiling" and "not
		# smiling", then set the label accordingly
		(notSmiling, smiling) = model.predict(roi)[0]
		label = "Smiling" if smiling > notSmiling else "Not Smiling"

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frameClone, label, (fX, fY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
			(0, 0, 255), 2)

	# show our detected faces along with smiling/not smiling labels
	cv2.imshow("Face", frameClone)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()