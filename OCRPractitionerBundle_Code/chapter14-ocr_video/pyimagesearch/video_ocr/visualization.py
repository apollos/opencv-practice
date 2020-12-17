# import the necessary packages
import numpy as np

class VideoOCROutputBuilder:
	def __init__(self, frame):
		# store the input frame dimensions
		self.maxW = frame.shape[1]
		self.maxH = frame.shape[0]

	def build(self, frame, card=None, ocr=None):
		# grab the input frame dimensions and  initialize the card
		# image dimensions along with the OCR image dimensions
		(frameH, frameW) = frame.shape[:2]
		(cardW, cardH) = (0, 0)
		(ocrW, ocrH) = (0, 0)

		# if the card image is not empty, grab its dimensions
		if card is not None:
			(cardH, cardW) = card.shape[:2]

		# similarly, if the OCR image is not empty, grab its
		# dimensions
		if ocr is not None:
			(ocrH, ocrW) = ocr.shape[:2]

		# compute the spatial dimensions of the output frame
		outputW = max([frameW, cardW, ocrW])
		outputH = frameH + cardH + ocrH

		# update the max output spatial dimensions found thus far
		self.maxW = max(self.maxW, outputW)
		self.maxH = max(self.maxH, outputH)

		# allocate memory of the output image using our maximum
		# spatial dimensions
		output = np.zeros((self.maxH, self.maxW, 3), dtype="uint8")

		# set the frame in the output image
		output[0:frameH, 0:frameW] = frame

		# if the card is not empty, add it to the output image
		if card is not None:
			output[frameH:frameH + cardH, 0:cardW] = card

		# if the OCR result is not empty, add it to the output image
		if ocr is not None:
			output[
				frameH + cardH:frameH + cardH + ocrH,
				0:ocrW] = ocr

		# return the output visualization image
		return output