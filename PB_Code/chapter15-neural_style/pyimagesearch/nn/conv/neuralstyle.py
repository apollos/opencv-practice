# import the necessary packages
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import cv2
import os

class NeuralStyle:
	def __init__(self, settings):
		# store the settings dictionary
		self.S = settings

		# grab the dimensions of the input image
		(w, h) = load_img(self.S["input_path"]).size
		self.dims = (h, w)

		# load content image and style images, forcing the dimensions
		# of our input image
		self.content = self.preprocess(settings["input_path"])
		self.style = self.preprocess(settings["style_path"])
		self.content = K.variable(self.content)
		self.style = K.variable(self.style)

		# allocate memory of our output image, then combine the
		# content, style, and output into a single tensor so they can
		# be fed through the network
		self.output = K.placeholder((1, self.dims[0],
			self.dims[1], 3))
		self.input = K.concatenate([self.content, self.style,
			self.output], axis=0)

		# load our model from disk
		print("[INFO] loading network...")
		self.model = self.S["net"](weights="imagenet",
			include_top=False, input_tensor=self.input)

		# build a dictionary that maps the *name* of each layer
		# inside the network to the actual layer *output*
		layerMap = {l.name: l.output for l in self.model.layers}

		# extract features from the content layer, then extract the
		# activations from the style image (index 0) and the output
		# image (index 2) -- these will serve as our style features
		# and output features from the *content* layer
		contentFeatures = layerMap[self.S["content_layer"]]
		styleFeatures = contentFeatures[0, :, :, :]
		outputFeatures = contentFeatures[2, :, :, :]

		# compute the feature reconstruction loss, weighting it
		# appropriately
		contentLoss = self.featureReconLoss(styleFeatures,
			outputFeatures)
		contentLoss *= self.S["content_weight"]

		# initialize our style loss along with the value used to
		# weight each style layer (in proportion to the total number
		# of style layers
		styleLoss = K.variable(0.0)
		weight = 1.0 / len(self.S["style_layers"])

		# loop over the style layers
		for layer in self.S["style_layers"]:
			# grab the current style layer and use it to extract the
			# style features and output features from the *style
			# layer*
			styleOutput = layerMap[layer]
			styleFeatures = styleOutput[1, :, :, :]
			outputFeatures = styleOutput[2, :, :, :]

			# compute the style reconstruction loss as we go
			T = self.styleReconLoss(styleFeatures, outputFeatures)
			styleLoss += (weight * T)

		# finish computing the style loss, compute the total
		# variational loss, and then compute the total loss that
		# combines all three
		styleLoss *= self.S["style_weight"]
		tvLoss = self.S["tv_weight"] * self.tvLoss(self.output)
		totalLoss = contentLoss + styleLoss + tvLoss

		# compute the gradients out of the output image with respect
		# to loss
		grads = K.gradients(totalLoss, self.output)
		outputs = [totalLoss]
		outputs += grads

		# the implementation of L-BFGS we will be using requires that
		# our loss and gradients be *two separate functions* so here
		# we create a Keras function that can compute both the loss
		# and gradients together and then return each separately
		# using two different class methods
		self.lossAndGrads = K.function([self.output], outputs)

	def preprocess(self, p):
		# load the input image (while resizing it to the desired
		# dimensions) and preprocess it
		image = load_img(p, target_size=self.dims)
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)

		# return the preprocessed image
		return image

	def deprocess(self, image):
		# reshape the image, then reverse the zero-centering by
		# *adding* back in the mean values across the ImageNet
		# training set
		image = image.reshape((self.dims[0], self.dims[1], 3))
		image[:, :, 0] += 103.939
		image[:, :, 1] += 116.779
		image[:, :, 2] += 123.680

		# clip any values falling outside the range [0, 255] and
		# convert the image to an unsigned 8-bit integer
		image = np.clip(image, 0, 255).astype("uint8")

		# return the deprocessed image
		return image

	def gramMat(self, X):
		# the gram matrix is the dot product between the input
		# vectors and their respective transpose
		features = K.permute_dimensions(X, (2, 0, 1))
		features = K.batch_flatten(features)
		features = K.dot(features, K.transpose(features))

		# return the gram matrix
		return features

	def featureReconLoss(self, styleFeatures, outputFeatures):
		# the feature reconstruction loss is the squared error
		# between the style features and output output features
		return K.sum(K.square(outputFeatures - styleFeatures))

	def styleReconLoss(self, styleFeatures, outputFeatures):
		# compute the style reconstruction loss where A is the gram
		# matrix for the style image and G is the gram matrix for the
		# generated image
		A = self.gramMat(styleFeatures)
		G = self.gramMat(outputFeatures)

		# compute the scaling factor of the style loss, then finish
		# computing the style reconstruction loss
		scale = 1.0 / float((2 * 3 * self.dims[0] * self.dims[1]) ** 2)
		loss = scale * K.sum(K.square(G - A))

		# return the style reconstruction loss
		return loss

	def tvLoss(self, X):
		# the total variational loss encourages spatial smoothness in
		# the output page -- here we avoid border pixels to avoid
		# artifacts
		(h, w) = self.dims
		A = K.square(X[:, :h - 1, :w - 1, :] - X[:, 1:, :w - 1, :])
		B = K.square(X[:, :h - 1, :w - 1, :] - X[:, :h - 1, 1:, :])
		loss = K.sum(K.pow(A + B, 1.25))

		# return the total variational loss
		return loss

	def transfer(self, maxEvals=20):
		# generate a random noise image that will serve as a
		# placeholder array, slowly modified as we run L-BFGS to
		# apply style transfer
		X = np.random.uniform(0, 255,
			(1, self.dims[0], self.dims[1], 3)) - 128

		# start looping over the desired number of iterations
		for i in range(0, self.S["iterations"]):
			# run L-BFGS over the pixels in our generated image to
			# minimize the neural style loss
			print("[INFO] starting iteration {} of {}...".format(
				i + 1, self.S["iterations"]))
			(X, loss, _) = fmin_l_bfgs_b(self.loss, X.flatten(),
				fprime=self.grads, maxfun=maxEvals)
			print("[INFO] end of iteration {}, loss: {:.4e}".format(
				i + 1, loss))

			# deprocess the generated image and write it to disk
			image = self.deprocess(X.copy())
			p = os.path.sep.join([self.S["output_path"],
				"iter_{}.png".format(i)])
			cv2.imwrite(p, image)

	def loss(self, X):
		# extract the loss value
		X = X.reshape((1, self.dims[0], self.dims[1], 3))
		lossValue = self.lossAndGrads([X])[0]

		# return the loss
		return lossValue

	def grads(self, X):
		# compute the loss and gradients
		X = X.reshape((1, self.dims[0], self.dims[1], 3))
		output = self.lossAndGrads([X])

		# extract and return the gradient values
		return output[1].flatten().astype("float64")