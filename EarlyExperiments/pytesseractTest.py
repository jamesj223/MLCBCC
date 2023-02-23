# OCR'd 100 files in 2 minutes. Not bad!

# import the necessary packages
import numpy as np
import os, argparse, imutils, cv2, pytesseract

from datetime import datetime

from tqdm import tqdm

try:
    from PIL import Image, ImageOps
except ImportError:
    import Image
 
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True,
#	help = "path to the image file")
#args = vars(ap.parse_args())

def recursiveFileSearch(folder, extensionList):
	files = []
	count = {}
	#count['Other'] = 0
	for root, dirnames, filenames in os.walk(folder):
		for filename in filenames:
			fullFilePathAndName = os.path.join(root, filename)
			for extension in extensionList:
				if filename.lower().endswith( tuple(extension) ):
					if fullFilePathAndName not in files:
						files.append(fullFilePathAndName)
						if extension in count:
							count[extension] += 1
						else:
							count[extension] = 1			
			if fullFilePathAndName not in files:
				#count["Other"] += 1
				extension = fullFilePathAndName.split(".")[-1]
				if extension in count:
					count[extension] += 1
				else:
					count[extension] = 1
				#if debug:
				#	print filename

	return files

startTime = datetime.now()

print "Start - " + str(datetime.now())

supportedImageExtensions = ['.jpg', '.jpeg']
#fileList = recursiveFileSearch("Development Set", supportedImageExtensions)
fileList = recursiveFileSearch("", supportedImageExtensions)

#def parallelHelper(file):

import re

pattern = re.compile(r"\W+")

for file in tqdm(fileList):

	image = Image.open(file)#cv2.imread(file)#, cv2.IMREAD_GRAYSCALE)#args["image"]

	# Invert
	#image = cv2.bitwise_not(image)

	allText = pytesseract.image_to_string(image).lower()

	wordsList = set( re.split(pattern, allText) )

	#mWords = for i in St.split():
    #if i.startswith("m"):
    #    print(i)

	print str(file) + " Character Count: " + str(len(allText))
	print str(file) + " Word Count: " + str(len(wordsList))
	print str(file) + " contains the word 'Marvel': " + str("marvel" in wordsList)
	print ""	
	#print wordsList

print "End - " + str(datetime.now())

print "Took: " + str( datetime.now() - startTime )
