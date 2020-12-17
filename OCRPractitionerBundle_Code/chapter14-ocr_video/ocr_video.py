# USAGE
# python ocr_video.py --input video/business_card.mp4

# import the necessary packages
from pyimagesearch.video_ocr import VideoOCROutputBuilder
from pyimagesearch.blur_detection import detect_blur_fft
from pyimagesearch.helpers import cleanup_text
from imutils.video import VideoStream
from imutils.perspective import four_point_transform
from pytesseract import Output
import pytesseract
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video (webcam will be used otherwise)")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video")
ap.add_argument("-c", "--min-conf", type=int, default=50,
	help="mininum confidence value to filter weak text detection")
args = vars(ap.parse_args())

# initialize our video OCR output builder used to easily visualize
# output to our screen
outputBuilder = None

# initialize our output video writer along with the dimensions of the
# output frame
writer = None
outputW = None
outputH = None

# create a named window for our output OCR visualization (a named
# window is required here so that we can automatically position it
# on our screen)
cv2.namedWindow("Output")

# initialize a boolean used to indicate if either a webcam or input
# video is being used
webcam = not args.get("input", False)

# if a video path was not supplied, grab a reference to the webcam
if webcam:
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# a webcam or a video file
	orig = vs.read()
	orig = orig if webcam else orig[1]

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if not webcam and orig is None:
		break

	# resize the frame and compute the ratio of the *new* width to
	# the *old* width
	frame = imutils.resize(orig, width=600)
	ratio = orig.shape[1] / float(frame.shape[1])

	# if our video OCR output builder is None, initialize it
	if outputBuilder is None:
		outputBuilder = VideoOCROutputBuilder(frame)

	# initialize our card and OCR output ROIs
	card = None
	ocr = None

	# convert the frame to grayscale and detect if the frame is
	# considered blurry or not
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	(mean, blurry) = detect_blur_fft(gray, thresh=15)

	# draw whether or not the frame is blurry
	color = (0, 0, 255) if blurry else (0, 255, 0)
	text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
	text = text.format(mean)
	cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
		0.7, color, 2)

	# only continue to process the frame for OCR if the image is
	# *not* blurry
	if not blurry:
		# blur the grayscale image slightly and then perform edge
		# detection
		blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
		edged = cv2.Canny(blurred, 75, 200)

		# find contours in the edge map and sort them by size in
		# descending order, keeping only the largest ones
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

		# initialize a contour that corresponds to the business card
		# outline
		cardCnt = None

		# loop over the contours
		for c in cnts:
			# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)

			# if our approximated contour has four points, then we
			# can assume we have found the outline of the business
			# card
			if len(approx) == 4:
				cardCnt = approx
				break

		# ensure that the business card contour was found
		if cardCnt is not None:
			# draw the outline of the business card on the frame so
			# we visually verify that the card was detected correctly
			cv2.drawContours(frame, [cardCnt], -1, (0, 255, 0), 3)

			# apply a four point perspective transform to the
			# *original* frame to obtain a top-down birds eye
			# view of the business card
			card = four_point_transform(orig,
				cardCnt.reshape(4, 2) * ratio)

			# allocate memory for our output OCR visualization
			ocr = np.zeros(card.shape, dtype="uint8")

			# swap channel ordering for the business card and OCR it
			rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
			results = pytesseract.image_to_data(rgb,
				output_type=Output.DICT)

			# loop over each of the individual text localizations
			for i in range(0, len(results["text"])):
				# extract the bounding box coordinates of the text
				# region from the current result
				x = results["left"][i]
				y = results["top"][i]
				w = results["width"][i]
				h = results["height"][i]

				# extract the OCR text itself along with the
				# confidence of the text localization
				text = results["text"][i]
				conf = int(results["conf"][i])

				# filter out weak confidence text localizations
				if conf > args["min_conf"]:
					# process the text by stripping out non-ASCII
					# characters
					text = cleanup_text(text)

					# if the cleaned up text is not empty, draw a
					# bounding box around the text along with the
					# text itself
					if len(text) > 0:
						cv2.rectangle(card, (x, y), (x + w, y + h),
							(0, 255, 0), 2)
						cv2.putText(ocr, text, (x, y - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5,
							(0, 0, 255), 1)

	# build our final video OCR output visualization
	output = outputBuilder.build(frame, card, ocr)

	# check if the video writer is None *and* an output video file
	# path was supplied
	if args["output"] is not None and writer is None:
		# grab the output frame dimensions and initialize our video
		# writer
		(outputH, outputW) = output.shape[:2]
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 27,
			(outputW, outputH), True)

	# if the writer is not None, we need to write the output video
	# OCR visualization to disk
	if writer is not None:
		# force resize the video OCR visualization to match the
		# dimensions of the output video
		outputFrame = cv2.resize(output, (outputW, outputH))
		writer.write(outputFrame)

	# show the output video OCR visualization
	cv2.imshow("Output", output)
	cv2.moveWindow("Output", 0, 0)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# if we are using a webcam, stop the camera video stream
if webcam:
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()