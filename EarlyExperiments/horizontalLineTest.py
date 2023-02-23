#!python2
import numpy as np
import os, argparse, imutils, cv2, pytesseract

from datetime import datetime

from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    import Image
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "path to the image file")
args = vars(ap.parse_args())

threshold = 10

# Try to find panel borders/gaps
# This is a fucking mess.
#def getHorizontalLineFeatures(File):
def takeOne(File):
	BW = cv2.imread(File, cv2.IMREAD_GRAYSCALE)

	(height, width, channels) = cv2.imread(File).shape

	hLinesWhite = 0
	hLinesBlack = 0

	#https://stackoverflow.com/questions/25642532/opencv-pointx-y-represent-column-row-or-row-column
	for i in tqdm( range(height) ):
		firstPixel = -1

		for j in range(width):
			#print "Line 31"
			currentPixel = BW[i,j]
			#print "BW[" + str(i) + "," + str(j) + "] : " + str(currentPixel)
			# Ignore prevPixel code on first loop
			if firstPixel >= 0:
				#print "Line 36"
				difference = abs( int(currentPixel)-int(firstPixel) )
				#print "Difference = " + str(difference)
				if difference <= threshold:
					#print "Current Pixel within threshold of first pixel in row"
					if (j == width-1):
						#print "End of Line Reached!"
						# End of Line Reached!
						# Increment relevant counter
						if (currentPixel <= 0 + threshold):
							hLinesBlack += 1
						elif (currentPixel > 255 - threshold):
							hLinesWhite += 1
				
				# Prev and current don't match, next line
				else:
					#print "Line 50"
					continue
			
			# Else first pixel not set, so set it.
			else:
				firstPixel = currentPixel

			# if current is not white or black, move to next line
			if (currentPixel >= 0 + threshold) or (currentPixel <= 255 - threshold):#not in [0,255]:
				continue

			#else:
				#print currentPixel
				#print "Line 57"
			#prevPixel = currentPixel
			#print "Line 60"

	print "hLinesWhite: " + str(hLinesWhite)
	print "hLinesBlack: " + str(hLinesBlack)  

	return (hLinesWhite, hLinesBlack)

# Rather than perform hundreds of pixel comparisons
# Sum each row, then perform perform 1 comparison on that value
def takeTwo(File):
	BW = cv2.imread(File, cv2.IMREAD_GRAYSCALE)

	(height, width, channels) = cv2.imread(File).shape

	hLinesWhite = 0
	hLinesBlack = 0

	threshold = 10

	# Calculate min value (black) and max value (white) 
	# If comic was true black (0 value) it would be 0, but seems like we need a threshold.
	# Likewise for true white (255) it would be 255 * width
	minValue = threshold * width

	maxValue = (255 - threshold) * width

	#https://stackoverflow.com/questions/25642532/opencv-pointx-y-represent-column-row-or-row-column
	for i in range(height): #tqdm( ):
		lineSum = np.sum( BW[ i, 0:-1 ] )
		judgement = ""
		if lineSum < minValue:
			#judgement = " - Black Line"
			hLinesBlack += 1
		elif lineSum > maxValue:
			#judgement = " - White Line"
			hLinesWhite += 1
		#print str(i) + " : " + str( lineSum ) + judgement

	print "hLinesWhite: " + str(hLinesWhite)
	print "hLinesBlack: " + str(hLinesBlack)  

	return (hLinesWhite, hLinesBlack)

#takeOne(args["image"])
takeTwo(args["image"])