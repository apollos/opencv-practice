# import necessary packages
import sensor
import time
import image

# import the lcd optionally
# to use the LCD, uncomment Lines 9, 24, 33, and 100
# and comment Lines 19 and 20 
#import lcd

# reset the camera
sensor.reset()

# sensor settings
sensor.set_pixformat(sensor.GRAYSCALE)

# non LCD settings
# comment the following lines if you are using the LCD
sensor.set_framesize(sensor.VGA)
sensor.set_windowing((640, 240))

# LCD settings
# uncomment this line to use the LCD with valid resolution
#sensor.set_framesize(sensor.QQVGA2)

# additional sensor settings
sensor.skip_frames(2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)

# initialize the LCD
# uncomment if you are using the LCD
#lcd.init()

# initialize the clock
clock = time.clock()

# barcode type lookup table
barcode_type = {
	image.EAN2: "EAN2",
	image.EAN5: "EAN5",
	image.EAN8: "EAN8",
	image.UPCE: "UPCE",
	image.ISBN10: "ISBN10",
	image.EAN13: "EAN13",
	image.ISBN13: "ISBN13",
	image.I25: "I25",
	image.DATABAR: "DATABAR",
	image.DATABAR_EXP: "DATABAR_EXP",
	image.CODABAR: "CODABAR",
	image.CODE39: "CODE39",
	image.PDF417: "PDF417",
	image.CODE93: "CODE93",
	image.CODE128: "CODE128"
}

def barcode_name(code):
	# if the code type is in the dictionary, return the value string
	if code.type() in barcode_type.keys():
		return barcode_type[code.type()]

	# otherwise return a "not defined" string
	return "NOT DEFINED"

# loop over frames and detect + decode barcodes
while True:
	# tick the clock for our FPS counter
	clock.tick()

	# grab a frame
	img = sensor.snapshot()

	# loop over standard barcodes that are detected in the image
	for code in img.find_barcodes():
		# draw a rectangle around the barcode
		img.draw_rectangle(code.rect(), color=127)

		# print information in the IDE terminal
		print("type: {}, quality: {}, payload: {}".format(
			barcode_name(code),
			code.quality(),
			code.payload()))

		# draw the barcode string on the screen similar to cv2.putText
		img.draw_string(10, 10, code.payload(), color=127)

	# loop over QR codes that are detected in the image
	for code in img.find_qrcodes():
		# draw a rectangle around the barcode
		img.draw_rectangle(code.rect(), color=127)

		# print information in the IDE terminal
		print("type: QR, payload: {}".format(code.payload()))

		# draw the barcode string on the screen similar to cv2.putText
		img.draw_string(10, 10, code.payload(), color=127)

	# display the image on the LCD
	# uncomment if you are using the LCD
	#lcd.display(img)

	# print the frames per second for debugging
	print("FPS: {}".format(clock.fps()))