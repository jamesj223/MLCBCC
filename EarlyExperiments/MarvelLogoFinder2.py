#!python2
# Example/Method 2

#https://stackoverflow.com/questions/7853628/how-do-i-find-an-image-contained-within-an-image

# import the necessary packages
import numpy as np
import argparse, imutils, cv2, pytesseract

from matplotlib import pyplot as plt

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
template = cv2.imread('marvel.jpg')
img2 = cv2.imread(image)

def imageSearchMethodComparison(Image):

    w, h = template.shape[:2]#[::-1]

    # All the 6 methods for comparison in a list
    # (methodString, fiftyPercentThreshold, seventyFivePercentThreshold)
    methods = [
        ('cv2.TM_CCOEFF', 530830296, 665854060),
        ('cv2.TM_CCOEFF_NORMED', 0.636270106, 0.816843629),
        ('cv2.TM_CCORR', 5466229248, 5452890880),
        ('cv2.TM_CCORR_NORMED', 0.958957106, 0.979280487),
        ('cv2.TM_SQDIFF', 432032077.5, 218196436.3),
        ('cv2.TM_SQDIFF_NORMED', 0.082749779, 0.041780502),
    ]

    for (methodString, fiftyPercentThreshold, seventyFivePercentThreshold) in methods:

        print methodString
        startTime = datetime.now()
        #print "Start - " + str(datetime.now())

        img = img2.copy()
        method = eval(methodString)

        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            confidence = min_val

            if confidence < seventyFivePercentThreshold:
                print "!!Confidence above Seventy Five Percent Threshold!!"
            elif confidence < fiftyPercentThreshold:
                print "!Confidence above Fifty Percent Threshold!"
            else:
                print "Confidence = " + str(confidence) + " (Lower is better?)"
        else:
            top_left = max_loc
            confidence = max_val

            if confidence > seventyFivePercentThreshold:
                print "!!Confidence above Seventy Five Percent Threshold!!"
            elif confidence > fiftyPercentThreshold:
                print "!Confidence above Fifty Percent Threshold!"
            else:
                print "Confidence = " + str(confidence) + " (Higher is better?)"

        #print "min_val, max_val : (" + str(min_val) + "," +str(max_val) + ")"

        bottom_right = (top_left[0] + w, top_left[1] + h)

        #cv2.rectangle(img,top_left, bottom_right, 255, 2)

        #plt.subplot(121),plt.imshow(res,cmap = 'gray')
        #plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        #plt.subplot(122),plt.imshow(img,cmap = 'gray')
        #plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        #plt.suptitle(methodString)

        #print "End - " + str(datetime.now())

        print "Took: " + str( datetime.now() - startTime )
        print ""

    #plt.show()

imageSearchMethodComparison(image)
