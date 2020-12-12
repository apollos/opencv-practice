# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K

class SRCNN:
	@staticmethod
	def build(width, height, depth):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# the entire SRCNN architecture consists of three CONV =>
		# RELU layers with *no* zero-padding
		model.add(Conv2D(64, (9, 9), kernel_initializer="he_normal",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(Conv2D(32, (1, 1), kernel_initializer="he_normal"))
		model.add(Activation("relu"))
		model.add(Conv2D(depth, (5, 5),
			kernel_initializer="he_normal"))
		model.add(Activation("relu"))

		# return the constructed network architecture
		return model