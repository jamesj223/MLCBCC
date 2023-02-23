#!python2
###############################################################################
# MachineLearningExperiment.py - Machine Learning Comic Book Cover Finding Experiment
# jamesj223

''' 

TODO

Create CSVs

    Development set - Done

    Training set
    Validation set
        Or just do the random selection/split thing for training...

    Full set
        Just a recursive list of every archive and containing image

Feature Set

    Archive name
    File name - Image
    Aspect ratio / Horizontal Resolution / Vertical Resolution
    Horizontal black / white lines 
    Number of each colour pixels
    "Average" colour
    Barcode or marvel logo
    Variant in file name
    # covers in archive name
    File position in archive
    # Words on page (using OCR)
        Look for key words? "Variant" specifically comes to mind

Misc Thoughts
    BW Threshold? How perfect does it need to be?

    Line threshold - How many different coloured pixels before we say it's not a continuous line

    Should we check for other coloured lines? what if BG isn't B or W

'''

###############################################################################
# Initialisation

import os, cPickle, gzip

import re
pattern = re.compile(r"\W+")

from datetime import datetime, timedelta

import imutils, cv2, pytesseract 
import numpy as np

from tqdm import tqdm

#Attempting parallel processing! :s
from multiprocessing import Pool

try:
    from PIL import Image
except ImportError:
    import Image

###############################################################################
# Config

supportedImageExtensions = ['.jpg', '.jpeg']

inputDirectory = ""

outputCSV = "zFeatureEngineeringOptimization"

# Clean flag, will delete pickles and recreate everything from scratch
clean = True

# Attempt Parallel Processing :s
parallel = True

# How many images to process
batchSize = 100

# Debug flag
debug = False

###############################################################################
# Classes

################################################################################
# Pickle Functions - With gzip

def save_obj(obj, name, protocol=2):
  with open('obj/'+ name + '.pkl', 'wb') as f:
      cPickle.dump(obj, f, protocol)

def load_obj(name ):
  with open('obj/' + name + '.pkl', 'rb') as f:
    return cPickle.load(f)

def saveAndZip(obj, name, protocol=2):
  filename = 'obj/' + name + '.pkz'
  file = gzip.GzipFile(filename, 'wb')
  file.write(cPickle.dumps(obj, protocol))
  file.close()

  return filename

def loadAndUnZip(name):
  filename = 'obj/' + name + '.pkz'
  file = gzip.GzipFile(filename, 'rb')
  buffer = ""
  while True:
          data = file.read()
          if data == "":
                  break
          buffer += data
  object = cPickle.loads(buffer)
  file.close()
  return object

# Returns pickle if exsits or False if it doesn't
def loadPickleIfExists( Name ):
    try:
      pickle = loadAndUnZip( Name )
    except:
      pickle = False
    return pickle

###############################################################################
# Functions

# Creates directory if it doesn't already exist
def createDirectoryIfNotExists(path):
  if not os.path.exists(path):
    os.makedirs(path)

def recursiveFileSearchByExtension(folder, extensionList):
    files = []
    count = {}
    #count['Other'] = 0
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            fullFilePathAndName = os.path.join(root, filename)
            for extension in extensionList:
                if filename.lower().endswith( tuple(extension) ):
                    if fullFilePathAndName not in files:
                        files.append( os.path.normpath( fullFilePathAndName ) )
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
                #   print filename

    return files

def makeCSV(filename, data):
    if not filename.endswith(".csv"):
        filename += ".csv"

    f = open(filename, 'wb')

    # Headers? - TODO
    headerLine = ",".join(
        (
        "fileName",
        "FNhasVariant",
        "height",
        "width",
        "hLinesBlack",
        "hLinesWhite",
        "numWhitePixels",
        "numBlackPixels",
        "OCRwordCount",
        "OCRhasVariant",
        "OCRhasMarvel",
        "IMGhasMarvelBool",
        "IMGhasMarvelScore"
        )
    )

    headerLine += "\n"
    f.write(headerLine)

    for line in data:
        csvLine = ""
        for thing in line:
            csvLine += str(thing).replace(',', '') + "," #Attempting to mitigate
                    #the "fix" from reateIndivualImageFeatureSet(File) ~line 254
        csvLine = csvLine[0:-1] + "\n"
        f.write(csvLine)

    f.close()

def clean_obj_folder():
    path = './' + 'obj'
    for filename in os.listdir(path):
        if filename.endswith('pkz'):
            f = os.path.join(path, filename)
            os.remove(f)

# CreateIndivualImageFeatureSet(Image) -> Return full image feature set for given image
# Excludes meta features (ArchiveName, ImageName) - these can be added outside this function
def createFeatureSet(FileList):
    
    featureSet = []

    # Normal
    if not parallel:
        for file in tqdm(FileList, ascii=True):
            fileName = os.path.basename(file)
            tqdm.write( fileName , end="" )
            featureSet.append( createIndivualImageFeatureSet(file) )

    # Parallel
    else: 
        p = Pool()
        # add an init function? to assign a pyteesseract instance to each thread
        featureSet += p.map(createIndivualImageFeatureSet, FileList)
        p.close()
        p.join()

    archiveName = "CBR"

    #File position in archive? Issue is current planned training set would be super biased towards pos[0] or pos[1]
        # If I just wanted to catch cbr's with mutliples at the start, ML wouldn't be necessary
        # I want to catch the covers out of the middle of collected books too.

    #Cover(s) in archive name - same issue with current training set bias

    return featureSet

def createIndivualImageFeatureSet(File):

    #print File
    fileName = os.path.basename(File)#.replace(',', '')#Maybe un-comment this before you make CSVs!
    print "Building feature set for: " + fileName

    wordsList = set( re.split(pattern, fileName.lower()) )

    FNhasVariant = "variant" in wordsList
    
    startTime = datetime.now()
    (height, width, channels) = cv2.imread(File).shape
    if not parallel: featureTiming["basic"] += ( datetime.now() - startTime )

    startTime = datetime.now()
    (hLinesBlack, hLinesWhite) = getHorizontalLineFeatures(File)
    if not parallel: featureTiming["hLines"] += ( datetime.now() - startTime )

    #Number of each colour pixels?
    startTime = datetime.now()
    (numWhitePixels, numBlackPixels) = getColourFeatures(File)
    if not parallel: featureTiming["bwPixels"] += ( datetime.now() - startTime )

    #"Average" colour - https://www.pyimagesearch.com/2014/03/03/charizard-explains-describe-quantify-image-using-feature-vectors/
    #>>> means = cv2.mean(image)
    #>>> means
    #(181.12238527002307, 199.18315040165433, 206.514296508391, 0.0)

    #Presence of Barcode or marvel logo - done kinda. 
    #Doesn't seem to be working well. Too many false positives and false negatives.
    #And computationally expensive, relative to other features

    #Variant in file name? Add this next
    
    startTime = datetime.now()
    (OCRwordCount, OCRhasVariant, OCRhasMarvel) = getOCRFeatures(File)
    if not parallel: featureTiming["tesseract"] += ( datetime.now() - startTime )

    startTime = datetime.now()
    (IMGhasMarvelBool, IMGhasMarvelScore) = getImageRecognitionFeatures(File)
    if not parallel: featureTiming["marvelLogo"] += ( datetime.now() - startTime )

    featureSet = (
        fileName,
        int(FNhasVariant),
        height, 
        width, 
        hLinesBlack, 
        hLinesWhite,
        numWhitePixels,
        numBlackPixels,
        OCRwordCount, 
        int(OCRhasVariant), 
        int(OCRhasMarvel),
        int(IMGhasMarvelBool),
        IMGhasMarvelScore
    )

    return featureSet

# Try to find panel borders/gaps
def getHorizontalLineFeatures(File):
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

    #print "hLinesWhite: " + str(hLinesWhite)
    #print "hLinesBlack: " + str(hLinesBlack)  

    return (hLinesWhite, hLinesBlack)

# Get number of black and white pixels
def getColourFeatures(File):
    
    BW = cv2.imread(File, cv2.IMREAD_GRAYSCALE)
    
    blackPixelCount = np.sum(BW == 0)
    
    whitePixelCount = np.sum(BW == 255)

    return (blackPixelCount, whitePixelCount)

# Use OCR to try to find words on the page.
def getOCRFeatures(File):

    allText = pytesseract.image_to_string(Image.open(File)).lower()

    wordsList = set( re.split(pattern, allText) )

    wordCount = len(wordsList)

    hasVariant = "variant" in wordsList

    hasMarvel = "marvel" in wordsList

    return (wordCount, hasVariant, hasMarvel)

# Red Mask Helper for Image Rec
def redMaskThing(Image):
    lower = np.array([0, 0, 150], dtype = "uint8")
    upper = np.array([50, 50, 255], dtype = "uint8")
    mask = cv2.inRange(Image, lower, upper)
    output = cv2.bitwise_and(Image, Image, mask = mask)
    return output

# Use Image/Template Matching to try and find features
# At this stage just the marvel logo
def getImageRecognitionFeatures(File):

    return (0,0)

    template = cv2.imread("marvel.jpg")
    template = redMaskThing(template)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    
    threshold = 10000000 # 11000000.0

    image = cv2.imread(File)
    gray = redMaskThing(image)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    found = None
 
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
 
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
  
        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (maxVal, maxLoc, r) = found

    return ( ( maxVal > threshold ), maxVal )

def OLDgetImageRecognitionFeatures(File):

    template = cv2.imread('marvel.jpg')

    w, h = template.shape[:2]#[::-1]

    methodString = 'cv2.TM_CCOEFF'
    #fiftyPercentThreshold = 530830296
    seventyFivePercentThreshold = 665854060

    img = cv2.imread(File)
    method = eval(methodString)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    #top_left = max_loc
    #confidence = max_val

    return ( (max_val>seventyFivePercentThreshold), max_val )


###############################################################################
# Main

def datetime_to_float(d):
    total_seconds =  (d).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds

# def float_to_datetime(fl):
#     return datetime.fromtimestamp(fl)

def percentage(part, whole):
    a = datetime_to_float(part)
    b = datetime_to_float(whole)
    return " - " + str( 100 * a/b ) + "%"

if __name__ == '__main__':   

    featureTiming = {}

    featureTiming["basic"] = timedelta(0)
    featureTiming["hLines"] = timedelta(0)
    featureTiming["bwPixels"] = timedelta(0)
    featureTiming["tesseract"] = timedelta(0)
    featureTiming["marvelLogo"] = timedelta(0)

    # create obj folder if not exists
    createDirectoryIfNotExists('obj')

    if clean:
        clean_obj_folder()

    ### File List
    fileList = loadPickleIfExists("fileList")

    if clean or (fileList is False):
        #print "No pickle found for fileList, creating from scratch"\
        if debug: print inputDirectory
        fileList = recursiveFileSearchByExtension(inputDirectory, supportedImageExtensions)

    saveAndZip(fileList, "fileList")
    ### End File List

    #### Feature Set for File List
    featureSet = loadPickleIfExists("featureSet")

    if clean or (featureSet is False):
        #print "No pickle found for featureSet, creating from scratch"
        startTime = datetime.now()
        featureSet = createFeatureSet( fileList[0:batchSize] )
        if parallel: 
            print "Parallel Feature Time: " +  str( datetime.now() - startTime )

    saveAndZip(featureSet, "featureSet")
    ### End Feature Set

    ### CSV
    makeCSV(outputCSV, featureSet)

    if not parallel:
        totalFeatureTime = timedelta(0)
        for thing in featureTiming.values():
            totalFeatureTime += thing

        print "Total feature time: " + str(totalFeatureTime)

        print "Basic image features took: " + str(featureTiming["basic"]) + percentage(featureTiming["basic"], totalFeatureTime)
        print "Horizontal line features took: " + str(featureTiming["hLines"]) + percentage(featureTiming["hLines"], totalFeatureTime)
        print "BW pixel count features took: " + str(featureTiming["bwPixels"]) + percentage(featureTiming["bwPixels"], totalFeatureTime)
        print "Tesseract features took: " + str(featureTiming["tesseract"]) + percentage(featureTiming["tesseract"], totalFeatureTime)
        print "Marvel logo detection features took: " + str(featureTiming["marvelLogo"]) + percentage(featureTiming["marvelLogo"], totalFeatureTime)