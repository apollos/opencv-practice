# USAGE
# python resize.py --image jemma.png --baseline baseline.png --output output.png

# import the necessary packages
from conf import sr_config as config
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import argparse
import PIL
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-b", "--baseline", required=True,
	help="path to baseline image")
ap.add_argument("-o", "--output", required=True,
	help="path to output image")
args = vars(ap.parse_args())

# load the pre-trained model
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# load the input image, then grab the dimensions of the input image
# and crop the image such that it tiles nicely
print("[INFO] generating image...")
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
w -= int(w % config.SCALE)
h -= int(h % config.SCALE)
image = image[0:h, 0:w]

# resize the input image using bicubic interpolation then write the
# baseline image to disk
lowW = int(w * (1.0 / config.SCALE))
lowH = int(h * (1.0 / config.SCALE))
highW = int(lowW * (config.SCALE / 1.0))
highH = int(lowH * (config.SCALE / 1.0))
scaled = np.array(Image.fromarray(image).resize((lowW, lowH),
	resample=PIL.Image.BICUBIC))
scaled = np.array(Image.fromarray(scaled).resize((highW, highH),
	resample=PIL.Image.BICUBIC))
cv2.imwrite(args["baseline"], scaled)

# allocate memory for the output image
output = np.zeros(scaled.shape)
(h, w) = output.shape[:2]

# slide a window from left-to-right and top-to-bottom
for y in range(0, h - config.INPUT_DIM + 1, config.LABEL_SIZE):
	for x in range(0, w - config.INPUT_DIM + 1, config.LABEL_SIZE):
		# crop ROI from our scaled image
		crop = scaled[y:y + config.INPUT_DIM,
			x:x + config.INPUT_DIM].astype("float32")

		# make a prediction on the crop and store it in our output
		# image
		P = model.predict(np.expand_dims(crop, axis=0))
		P = P.reshape((config.LABEL_SIZE, config.LABEL_SIZE, 3))
		output[y + config.PAD:y + config.PAD + config.LABEL_SIZE,
			x + config.PAD:x + config.PAD + config.LABEL_SIZE] = P

# remove any of the black borders in the output image caused by the
# padding, then clip any values that fall outside the range [0, 255]
output = output[config.PAD:h - ((h % config.INPUT_DIM) + config.PAD),
	config.PAD:w - ((w % config.INPUT_DIM) + config.PAD)]
output = np.clip(output, 0, 255).astype("uint8")

# write the output image to disk
cv2.imwrite(args["output"], output)