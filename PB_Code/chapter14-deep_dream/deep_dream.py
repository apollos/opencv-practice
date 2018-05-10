# USAGE
# python deep_dream.py --image jp.jpg --output dream.png

# import the necessary packages
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
from scipy import ndimage
import numpy as np
import argparse
import cv2

def preprocess(p):
	# load the input image, convert it to a Keras-compatible array,
	# expand the dimensions so we can pass it through the model, and
	# then finally preprocess it for input to the Inception network
	image = load_img(p)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = preprocess_input(image)

	# return the preprocessed image
	return image

def deprocess(image):
	# we are using "channels last" ordering so ensure the RGB
	# channels are the last dimension in the matrix
	image = image.reshape((image.shape[1], image.shape[2], 3))

	# "undo" the preprocessing done for Inception to bring the image
	# back into the range [0, 255]
	image /= 2.0
	image += 0.5
	image *= 255.0
	image = np.clip(image, 0, 255).astype("uint8")

	# we have been processing images in RGB order; however, OpenCV
	# assumes images are in BGR order
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	# return the deprocessed image
	return image

def resize_image(image, size):
	# resize the image
	resized = np.copy(image)
	resized = ndimage.zoom(resized,
		(1, float(size[0]) / resized.shape[1],
		float(size[1]) / resized.shape[2], 1), order=1)

	# return the resized image
	return resized

def eval_loss_and_gradients(X):
	# fetch the loss and gradients given the input
	output = fetchLossGrads([X])
	(loss, G) = (output[0], output[1])

	# return a tuple of the loss and gradients
	return (loss, G)

def gradient_ascent(X, iters, alpha, maxLoss=-np.inf):
	# loop over our number of iterations
	for i in range(0, iters):
		# compute the loss and gradient
		(loss, G) = eval_loss_and_gradients(X)

		# if the loss is greater than the max loss, break from the
		# loop early to prevent strange effects
		if loss > maxLoss:
			break

		# take a step
		print("[INFO] Loss at {}: {}".format(i, loss))
		X += alpha * G

	# return the output of gradient ascent
	return X

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output dreamed image")
args = vars(ap.parse_args())

# define the dictionary that includes (1) the layers we are going
# to use for the dream and (2) their respective weights (i.e., the
# larger the weight, the more the layer contributes to the dream)
LAYERS = {
	"mixed2": 2.0,
	"mixed3": 0.5,
}

# define the number of octaves, octave scale, alpha (step for
# gradient ascent) number of iterations, and max loss -- tweaking
# these values will produce different dreams
NUM_OCTAVE = 3
OCTAVE_SCALE = 1.4
ALPHA = 0.001
NUM_ITER = 50
MAX_LOSS = 10.0

# indicate that Keras *should not* be update the weights of any
# layer during the deep dream
K.set_learning_phase(0)

# load the (pre-trained) Inception model from disk, then grab a
# reference variable to the input tensor of the model (which we'll
# then be using to perform our CNN hallucination)
print("[INFO] loading inception network...")
model = InceptionV3(weights="imagenet", include_top=False)
dream = model.input

# define our loss value, then build a dictionary that maps the
# *name* of each layer inside of Inception to the actual *layer*
# object itself -- we'll need this mapping when building the loss
# of the dream
loss = K.variable(0.0)
layerMap = {layer.name: layer for layer in model.layers}

# loop over the layers that will be utilized in the dream
for layerName in LAYERS:
	# grab the output of the layer we will use for dreaming, then add
	# the L2-norm of the features to the layer to the loss (we use
	# array slicing here to avoid border artifacts caused by border
	# pixels)
	x = layerMap[layerName].output
	coeff = LAYERS[layerName]
	scaling = K.prod(K.cast(K.shape(x), "float32"))
	loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

# compute the gradients of the dream with respect to loss and then
# normalize
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

# we now need to define a function that can retrieve the value of the
# loss and gradients given an input image
outputs = [loss, grads]
fetchLossGrads = K.function([dream], outputs)

# load and preprocess the input image, then grab the (original) input
# height and width
image = preprocess(args["image"])
dims = image.shape[1:3]

# in order to perform deep dreaming we need to build multiple scales
# of the input image (i.e., set of images at lower and lower
# resolutions) -- this list stores the spatial dimensions that we
# will be resizing our input image to
octaveDims = [dims]

# here we loop over the number of octaves (resolutions) we are going
# to generate
for i in range(1, NUM_OCTAVE):
	# compute the spatial dimensions (i.e., width and height) for the
	# current octave, then update the dimensions list
	size = [int(d / (OCTAVE_SCALE ** i)) for d in dims]
	octaveDims.append(size)

# reverse the octave dimensions list so that the *smallest*
# dimensions are at the *front* of the list
octaveDims = octaveDims[::-1]

# clone the original image and then create a resized input image that
# matches the smallest dimensions
orig = np.copy(image)
shrunk = resize_image(image, octaveDims[0])

# loop over the ocative dimensions from smallest to largest
for (o, size) in enumerate(octaveDims):
	# resize the image and then apply gradient ascent
	print("[INFO] starting octave {}...".format(o))
	image = resize_image(image, size)
	image = gradient_ascent(image, iters=NUM_ITER, alpha=ALPHA,
		maxLoss=MAX_LOSS)

	# to compute the lost detail we need two images: (1) the shrunk
	# image that has been upscaled to the current octave and (2) the
	# original image that has been downscaled to the current octave
	upscaled = resize_image(shrunk, size)
	downscaled = resize_image(orig, size)

	# the lost detail is computed via a simple subtraction which we
	# immediately back in to the image we applied gradient ascent to
	lost = downscaled - upscaled
	image += lost

	# make the original image be the new shrunk image so we can
	# repeat the process
	shrunk = resize_image(orig, size)

# deprocess our dream and save it to disk
image = deprocess(image)
cv2.imwrite(args["output"], image)