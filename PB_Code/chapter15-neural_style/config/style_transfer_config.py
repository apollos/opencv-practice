# import the necessary packages
import os

# define the content layer from which feature maps will be extracted
contentLayers = ["block4_conv2"]

# define the list of style layer blocks from our pre-trained CNN
styleLayers = [
	"block1_conv1",
	"block2_conv1",
	"block3_conv1",
	"block4_conv1",
	"block5_conv1"
]

# define the style weight, content weight, and total-variational
# loss weight, respectively (these are the values you'll want to
# tune to generate new style transfers)
styleWeight = 1.0
contentWeight = 1e4
tvWeight = 20.0

# define the number of epochs to train for along with the steps
# per each epoch
epochs = 15
stepsPerEpoch = 100

# define the path to the input content image, input style image,
# final output image, and path to the directory that will store
# the intermediate outptus
contentImage = os.path.sep.join(["inputs", "jp.jpg"])
styleImage = os.path.sep.join(["inputs", "mcescher.jpg"])
finalImage = "final.png"
intermOutputs = "intermediate_outputs"