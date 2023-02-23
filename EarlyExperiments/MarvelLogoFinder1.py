#!python2
# Example/Method 1

#https://stackoverflow.com/questions/7853628/how-do-i-find-an-image-contained-within-an-image

# import the necessary packages
import numpy as np
import argparse, imutils, cv2, pytesseract

from datetime import datetime

try:
    from PIL import Image
except ImportError:
    import Image
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "path to the image file")
args = vars(ap.parse_args())

image = args["image"]

methods = [
	("cv2.TM_SQDIFF_NORMED",cv2.TM_SQDIFF_NORMED),
]

def findMarvel(Image):

	# Start of image finding code
	method = cv2.TM_SQDIFF_NORMED

	# Read the images from the file
	small_image = cv2.imread('marvel.jpg')
	large_image = cv2.imread(Image)

	result = cv2.matchTemplate(small_image, large_image, method)

	# We want the minimum squared difference
	mn,_,mnLoc,_ = cv2.minMaxLoc(result)

	# Draw the rectangle:
	# Extract the coordinates of our best match
	MPx,MPy = mnLoc

	# Step 2: Get the size of the template. This is the same size as the match.
	trows,tcols = small_image.shape[:2]

	# Step 3: Draw the rectangle on large_image
	#cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,255,0),2)

	# Display the original image with the rectangle around the match.
	#cv2.imshow('output',large_image)

	# The image is only displayed if we call this
	#cv2.waitKey(0)

	print result

	return result

startTime = datetime.now()

print "Start - " + str(datetime.now())

findMarvel(image)

print "End - " + str(datetime.now())

print "Took: " + str( datetime.now() - startTime )