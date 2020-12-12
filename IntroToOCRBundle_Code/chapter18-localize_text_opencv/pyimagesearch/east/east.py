# import the necessary packages
import numpy as np

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
EAST_OUTPUT_LAYERS = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

def decode_predictions(scores, geometry, minConf=0.5):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# grab the confidence score for the current detection
			score = float(scoresData[x])

			# if our score does not have sufficient probability,
			# ignore it
			if score < minConf:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# use the offset and angle of rotation information to
			# start the calculation of the rotated bounding box
			offset = ([
				offsetX + (cos * xData1[x]) + (sin * xData2[x]),
				offsetY - (sin * xData1[x]) + (cos * xData2[x])])

			# derive the top-right corner and bottom-right corner of
			# the rotated bounding box
			topLeft = ((-sin * h) + offset[0], (-cos * h) + offset[1])
			topRight = ((-cos * w) + offset[0], (sin * w) + offset[1])

			# compute the center (x, y)-coordinates of the rotated
			# bounding box
			cX = 0.5 * (topLeft[0] + topRight[0])
			cY = 0.5 * (topLeft[1] + topRight[1])

			# our rotated bounding box information consists of the
			# center (x, y)-coordinates of the box, the width and
			# height of the box, as well as the rotation angle
			box = ((cX, cY), (w, h), -1 * angle * 180.0 / np.pi)

			# update our detections and confidences lists
			rects.append(box)
			confidences.append(score)

	# return a 2-tuple of the bounding boxes and associated
	# confidences
	return (rects, confidences)