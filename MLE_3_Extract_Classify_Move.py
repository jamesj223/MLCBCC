#!python2
###############################################################################
# MachineLearningExperiment.py - Machine Learning Comic Book Cover Finding Experiment
# jamesj223

###############################################################################
# Initialisation

from datetime import datetime

import os, sys, string, shutil, cPickle

import rarfile

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from MLE_1_Feature_Engineering import createFeatureSet

###############################################################################
# Config

# Mac
#tempDir = ""

#cbrDir = tempDir + "/1_InputFiles"
#extractDir = tempDir + "/2_Extract"
#flattenedDir = tempDir + "/3_Flattened"

#outputDir = ""

# Windows
tempDir = ""

cbrDir = tempDir + "\\1_InputFiles"
extractDir = tempDir + "\\2_Extract"
flattenedDir = tempDir + "\\3_Flattened"

outputDir = tempDir + "\\4_Output"

# Misc
supportedComicExtensions = ['.cbr']#, '.cbz']#
supportedImageExtensions = ['.jpg', '.jpeg']

#
verbose = True

classifier = 'GradientBoostingClassifier'#RandomForestClassifier

###############################################################################
# Classes

###############################################################################
# Functions

def save_obj(obj, name, protocol=2):
  with open('obj/'+ name + '.pkl', 'wb') as f:
      cPickle.dump(obj, f, protocol)

def load_obj(name ):
  with open('obj/' + name + '.pkl', 'rb') as f:
    return cPickle.load(f)

# Removed file extension counter stuff from this version of the function
def recursiveFileSearchByExtension(folder, extensionList):
    files = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            fullFilePathAndName = os.path.join(root, filename)
            for extension in extensionList:
                if filename.lower().endswith( extension ):
                    if fullFilePathAndName not in files:
                        files.append(fullFilePathAndName)           
    return files

def extractAllToSeparateFolders(baseFolder, archiveList):
    count = 0
    for archive in archiveList:
        rf = rarfile.RarFile(archive)
        folder = baseFolder + os.path.sep + "temp" + str(count)
        rf.extractall(path=folder)
        count += 1

# Helper for flattenAndRenameDuplicates
def recursiveFileSearch(path):
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for name in files:
                yield os.path.join(root, name)
    else:
        yield path

def flattenAndRenameDuplicates(fromdir, destination):

    dest = destination + os.path.sep
    
    for f in recursiveFileSearch(fromdir):
        filename = string.split(f, os.path.sep)[-1]
        if os.path.isfile(dest+filename):
            filename = f.replace(fromdir,"",1).replace(os.path.sep,"_")
        os.rename(f, dest+filename)
        #shutil.copy(f, dest+filename)

def cleanFolder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

###############################################################################
# Main

if __name__ == '__main__':   

    startTime = datetime.now()

    print("Start - " + str(datetime.now()))
    print("")

    ###########################################################################
    ### Code Starts Here

    # Cleanup MLE_Temp Folders
    if verbose: print "Cleaning up folders"
    cleanFolder(extractDir)
    cleanFolder(flattenedDir)

    # Create list of cbr/cbz archives
    if verbose: print "Creating list of cbr/cbz archives"
    fileList1 = recursiveFileSearchByExtension(cbrDir, supportedComicExtensions)

    #for file in fileList1:

    # Extract each archive to a separate folder
    if verbose: print "Extracting archives to separate folders"
    extractAllToSeparateFolders(extractDir, fileList1)#[file])#

    # Flatten folder structure and rename duplicates
    if verbose: print "Flattened extracted files and renaming duplicates"
    flattenAndRenameDuplicates(extractDir, flattenedDir)

    # Get new fileList for flattened folder
    if verbose: print "Getting flattened file list"
    fileList2 = recursiveFileSearchByExtension(flattenedDir, supportedImageExtensions)

    # Build featureSet for flattened fileList
    if verbose: print "Building feature set for flattened file list"
    featureSet = createFeatureSet(fileList2)

    # TEMP Save featureSet pickle
    if verbose: print "Saving feature set"
    save_obj(featureSet, "MLE_3_TempFeatureSet")

    # TEMP Load featureSet pickle
    #if verbose: print "Loading feature set"
    #featureSet = load_obj("MLE_3_TempFeatureSet")

    # Load Trained Classifier from Pickle
    if verbose: print "Loading trained classifier: " + classifier
    clf = load_obj(classifier)

    # Read "CSV" of files/features into pandas
    #pandaCSV = pd.read_csv(featureSet)
    if verbose: print "Load feature set into pandas"
    labels = [ "fileName", "FNhasVariant", "height", "width", "hLinesBlack", "hLinesWhite", "numWhitePixels", "numBlackPixels", "OCRwordCount", "OCRhasVariant", "OCRhasMarvel", "IMGhasMarvelBool", "IMGhasMarvelScore" ]
    pandaCSV = pd.DataFrame(data=featureSet, columns=labels)

    # Iterate over pandaCSV and apply classifier
    if verbose: print "Iterate over pandaCSV and apply classifier"
    matches = []
    for line in pandaCSV.iterrows():
        fileName = line[1].pop('fileName')
        #print fileName

        clf_input = line[1].to_numpy(copy=True).reshape(1, -1)
        #print clf_input
        
        clf_output = clf.predict(clf_input)
        #print clf_output

        if clf_output:
            matches.append(fileName)
            print fileName + " classified as cover!"

    # Move winners to MLE_Output
    if verbose: print "Moving matches to output folder"
    for match in matches:
        print "Moving from: " + flattenedDir + os.path.sep + match + " to " + outputDir + os.path.sep + match
        os.rename(flattenedDir + os.path.sep + match, outputDir + os.path.sep + match)

    # Cleanup MLE_Temp Folders
    if verbose: print "Cleaning up folders"
    cleanFolder(extractDir)
    cleanFolder(flattenedDir) # Temporarily disable this to manually search for false negatives.


    ### Code Ends Here
    ###########################################################################

    print("End - " + str(datetime.now()))

    print("Took: " + str( datetime.now() - startTime ))
