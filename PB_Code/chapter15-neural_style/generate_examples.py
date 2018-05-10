# USAGE
# python generate_examples.py

# import the necessary packages
from keras.applications import VGG19
from pyimagesearch.nn.conv import NeuralStyle
from imutils import paths
import json
import os

# initialize the dictionary of completed runs
completed = {}

# if the completed dictionary exists, load it
if os.path.exists("completed.json"):
	completed = json.loads(open("completed.json", "r").read())

# grab the set of example images
imagePaths = list(paths.list_images("inputs"))

# initialize the set of parameters/filenames
PARAMS = [
	"cw_1.0-sw_100.0-tvw_10.0",
	"cw_1.0-sw_1000.0-tvw_10.0",
	"cw_1.0-sw_100.0-tvw_100.0",
	"cw_1.0-sw_1000.0-tvw_1000.0",
	"cw_10.0-sw_100.0-tvw_10.0",
	"cw_10.0-sw_10.0-tvw_1000.0",
	"cw_10.0-sw_1000.0-tvw_1000.0",
	"cw_50.0-sw_10000.0-tvw_100.0",
	"cw_100.0-sw_1000.0-tvw_100.0"
]

# initialize the base dictionary
SETTINGS = {
	# initialize the path to the input (i.e., content) image,
	# style image, and path to the output directory
	"input_path": None,
	"style_path": None,
	"output_path": None,

	# define the CNN to be used style transfer, along with the
	# set of content layer and style layers, respectively
	"net": VGG19,
	"content_layer": "block4_conv2",
	"style_layers": ["block1_conv1", "block2_conv1",
		"block3_conv1", "block4_conv1", "block5_conv1"],

	# store the content, style, and total variation weights,
	# respectively
	"content_weight": None,
	"style_weight": None,
	"tv_weight": None,

	# number of iterations
	"iterations": 50,
}

# loop over the input images
for inputPath in imagePaths:
	for stylePath in imagePaths:
		# if the two paths are equal, ignore them
		if inputPath == stylePath:
			continue

		# loop over the parameters
		for param in PARAMS:
			# parse out the content weight, style weight, and total
			# variation weight from the string
			P = param.split("-")
			grid = {
				"content_weight": float(P[0].replace("cw_", "")),
				"style_weight": float(P[1].replace("sw_", "")),
				"tv_weight": float(P[2].replace("tvw_", "")),
			}

			# parse the filenames
			inputFilename = inputPath[inputPath.rfind("/") + 1:]
			inputFilename = inputFilename[:inputFilename.rfind(".")]
			styleFilename = stylePath[stylePath.rfind("/") + 1:]
			styleFilename = styleFilename[:styleFilename.rfind(".")]

			# construct the path to the output file
			p = "_".join([inputFilename, styleFilename, param])
			p = "{}.png".format(p)
			p = os.path.sep.join(["outputs", p])

			# update the settings dictionary
			SETTINGS["input_path"] = inputPath
			SETTINGS["style_path"] = stylePath
			SETTINGS["output_path"] = p
			SETTINGS["content_weight"] = grid["content_weight"]
			SETTINGS["style_weight"] = grid["style_weight"]
			SETTINGS["tv_weight"] = grid["tv_weight"]

			# build the key to the completed dictionary
			k = "{}_{}_{}".format(inputFilename, styleFilename, param)

			# if we have already performed this experiment, skip it
			if k in completed.keys():
				print("[INFO] skipping: {}".format(k))
				continue

			# perform neural style transfer with the current settings
			print("[INFO] starting: {}".format(k))
			ns = NeuralStyle(SETTINGS)
			ns.transfer()

			# indicate that the transfer completed successfully
			completed[k] = True

			# write the dictionary back out to disk
			print("[INFO] finished: {}".format(k))
			f = open("completed.json", "w")
			f.write(json.dumps(completed))
			f.close()