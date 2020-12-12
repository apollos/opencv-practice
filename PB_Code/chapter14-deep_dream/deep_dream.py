# USAGE
# python deep_dream.py --image jp.jpg --output dream.png

# import the necessary packages
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2

def loadImage(imagePath, width=350):
	# load the image from disk, resize it, swap the color channels
	# from BGR to RGB, and create a NumPy array of image pixel values
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=width)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = np.array(image)

	# return the image pixel array
	return image

def deprocess(image):
	# "undo" the preprocessing done for Inception and cast the pixel
	# values to integers
	image = 255 * (image + 1.0) 
	image /= 2.0
	image = tf.cast(image, tf.uint8)

	# return the deprocessed image
	return image

def calculateLoss(image, model):
	# add a batch dimension to the image and grab the activations
	# from specified layers of the Inception network after performing
	# a forward pass
	image = tf.expand_dims(image, axis=0)
	layerActivations = model(image)

	# initialize a list to store the intermediate losses
	losses = []
	
	# iterate over the layer activations
	for act in layerActivations:
		# compute the mean of each activation and append it to the
		# losses list list
		loss = tf.reduce_mean(act)
		losses.append(loss)

	# return the sum of the losses
	return tf.reduce_sum(losses)

@tf.function
def deepDream(model, image, stepSize, eps=1e-8):
	# instruct TenorFlow to record gradients
	with tf.GradientTape() as tape:
		# keep track of the image to calculate gradients and calculate
		# the loss yielded by the model
		tape.watch(image)
		loss = calculateLoss(image, model)
	
	# calculate the gradients of the loss with respect to the image
	# and normalize the gradients 
	gradients = tape.gradient(loss, image)
	gradients /= tf.math.reduce_std(gradients) + eps 
	
	# adjust the image with the normalized gradients and clip its
	# pixel values to the range [-1, 1]
	image = image + (gradients * stepSize)
	image = tf.clip_by_value(image, -1, 1)

	# return a tuple of the loss along with the updated image
	return (loss, image)

def runDeepDreamModel(model, image, iterations=100, stepSize=0.01):
	# preprocess the image for input to the Inception network
	image = preprocess_input(image)

	# loop for the given number of iterations
	for iteration in range(iterations):
		# employ our dreaming model to retrieve the loss along with
		# the updated image
		(loss, image) = deepDream(model, image, stepSize)
		
		# log the losses after a fixed interval
		if iteration % 25 == 0:
			print ("[INFO] iteration {}, loss {}".format(iteration,
				loss))

	# return the deprocessed image
	return deprocess(image)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output dreamed image")
args = vars(ap.parse_args())

# define the layers we are going to use for the dream
names = ["mixed3", "mixed5"]

# define the octave scale and number of octaves (tweaking these values
# will produce different output dreams)
OCTAVE_SCALE = 1.3
NUM_OCTAVES = 3

# load the input image
print("[INFO] loading image...")
originalImage = loadImage(args["image"])

# load the pre-trained Inception model from disk
print("[INFO] loading inception network...")
baseModel = InceptionV3(include_top=False, weights="imagenet")

# construct our dreaming model
layers = [baseModel.get_layer(name).output for name in names]
dreamModel = tf.keras.Model(inputs=baseModel.input, outputs=layers)

# convert the image to a TensorFlow constant for better performance,
# grab the first two dimensions of the image, and cast them to float
image = tf.constant(originalImage)
baseShape = tf.cast(tf.shape(image)[:-1], tf.float32)

# loop over the number of octaves (resolutions) we are going
# to generate
for n in range(NUM_OCTAVES):
	# compute the spatial dimensions (i.e., width and height) for the
	# current octave and cast them to integers
	print("[INFO] starting octave {}".format(n))
	newShape = tf.cast(baseShape * (OCTAVE_SCALE ** n), tf.int32)

	# resize the image with newly computed shape, convert it to its
	# NumPy variant, and run it through our dreaming model
	image = tf.image.resize(image, newShape).numpy()
	image = runDeepDreamModel(model=dreamModel, image=image, 
		iterations=200, stepSize=0.001)

# convert the final image to a NumPy array and save it to disk
finalImage = np.array(image)
Image.fromarray(finalImage).save(args["output"])