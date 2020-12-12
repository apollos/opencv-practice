# import necessary packages
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg19 import preprocess_input
from PIL import Image
import tensorflow as tf 
import numpy as np

class NeuralStyle(Model):
	def __init__(self, styleLayers, contentLayers):
		# call the parent constructor
		super(NeuralStyle, self).__init__()

		# construct our network with the given set of layers
		self.vgg =  self.vggLayers(styleLayers + contentLayers)

		# store the style layers, content layers, the number of style
		# layers, then set our network to non-trainable (if it is not
		# set already)
		self.styleLayers = styleLayers
		self.contentLayers = contentLayers
		self.numStyleLayers = len(styleLayers)
		self.vgg.trainable = False

	def call(self, inputs):
		# scale the pixel values of the image back to the [0, 255]
		# range and preprocess them
		inputs = inputs * 255.0
		preprocessedInput = preprocess_input(inputs)

		# run the preprocessed image through our network and grab the
		# style and content outputs
		outputs = self.vgg(preprocessedInput)
		(styleOutputs, contentOutputs) = (
			outputs[:self.numStyleLayers], 
			outputs[self.numStyleLayers:])

		# compute the gram matrix between the  different style outputs
		styleOutputs = [self.gramMatrix(styleOutput)
			for styleOutput in styleOutputs]

		# loop over the content layers (and their corresponding
		# outputs) and prepare a dictionary
		contentDict = {contentName:value 
			for contentName, value 
			in zip(self.contentLayers, contentOutputs)}

		# loop over the style layers (and their corresponding outputs)
		# and prepare a dictionary
		styleDict = {styleName:value
			for styleName, value
			in zip(self.styleLayers, styleOutputs)}
		
		# return a dictionary containing the style features, and the
		# content features
		return {"content": contentDict, "style": styleDict}
	
	@staticmethod
	def vggLayers(layerNames):
		# load our model from disk and set it non-trainable
		vgg = VGG19(include_top=False, weights="imagenet")
		vgg.trainable = False
	
		# construct a list of outputs of the specified layers, and then 
		# create the model
		outputs = [vgg.get_layer(name).output for name in layerNames]
		model = Model([vgg.input], outputs)

		# return the model
		return model
	
	@staticmethod
	def gramMatrix(inputTensor):
		# the gram matrix is the dot product between the input vectors
		# and their respective transpose
		result = tf.linalg.einsum("bijc,bijd->bcd",
			inputTensor, inputTensor)
		inputShape = tf.shape(inputTensor)
		locations = tf.cast(inputShape[1] * inputShape[2], 
			tf.float32)

		# return normalized gram matrix
		return (result / locations)

	@staticmethod
	def styleContentLoss(outputs, styleTargets, contentTargets,
		styleWeight, contentWeight):
		# extract the style and content outputs respectively
		styleOutputs = outputs["style"]
		contentOutputs = outputs["content"]

		# iterate over each of the style layers, grab their outputs,
		# and determine the mean-squared error with respect to the
		# original style content
		styleLoss = [tf.reduce_mean((
			styleOutputs[name] -  styleTargets[name]) ** 2)
			for name in styleOutputs.keys()]

		# add the individual style layer losses and normalize them
		styleLoss = tf.add_n(styleLoss)
		styleLoss *= styleWeight

		# iterate over each content layers, grab their outputs, and
		# determine the mean-squared error with respect to the
		# original  image content
		contentLoss = [tf.reduce_mean((contentOutputs[name] - 
			contentTargets[name]) ** 2) 
			for name in contentOutputs.keys()]

		# add the indvidual content layer losses and normalize them
		contentLoss = tf.add_n(contentLoss)
		contentLoss *= contentWeight

		# add the final style and content losses 
		loss = styleLoss + contentLoss

		# return the combined loss
		return loss

	@staticmethod
	def clipPixels(image):
		# clip any pixel values in the image falling outside the
		# range [0, 1] and return the image
		return tf.clip_by_value(image, 
			clip_value_min=0.0, 
			clip_value_max=1.0)

	@staticmethod
	def tensorToImage(tensor):
		# scale pixels back to the range [0, 255] and convert the
		# the data type of the pixels to integer
		tensor = tensor * 255
		tensor = np.array(tensor, dtype=np.uint8)

		# remove the batch dimension from the image if it is
		# present
		if np.ndim(tensor) > 3:
			tensor = tensor[0]

		# return the image in a PIL format
		return Image.fromarray(tensor)