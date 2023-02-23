# Machine Learning Comic Book Cover Classifier

## Overview

A series of scripts to use machine learning to find and extract covers from comic books. 

Most comic book files will have the cover as the first page, but often they will have multiple covers. Sometimes these are included at the start, sometimes at the end, and sometimes they are spread throughout the book. The goal of this project is to be able to run it against a directory full of comic book files, and have it extract all of the images it thinks are covers, so that they can be used to generate a cool collage (see the examples section below)

## Examples

Collages were built using [John's Background Switcher](https://johnsad.ventures/software/backgroundswitcher/)

Added some manually selected non-cover pages as well to give it a bit more variety.

![Example1](/Examples/1.jpg)
![Example2](/Examples/10.jpg)
![Example3](/Examples/21.jpg)

## Feature Engineering

MLE_1_Feature_Engineering.py is the first main file. Given a folder, recursively search through it for comic files (cbr/cbz) and build out a feature set for each file.

The features we are using are as follows:

- File Name
- Whether the file name contains "Variant"
- Image Height
- Image Width
- Number of continuous horizontal black lines in the image
- Number of continuous horizontal white lines in the image
- Number of white pixels in the image
- Number of black pixels in the image
- OCR word count for the image 
- Whether the OCR found the word "Variant"
- Whether the OCR found the word "Marvel"
- Whether OpenCV thinks it saw the Marvel Logo,
- OpenCV confident score it seeing the Marvel Logo

## Training

## Testing

## Early Experiments/Testing

## Misc Benchmarking

Comparison of different classifiers and how they performed against various training sets

![Comparison1](/Images/ClassifierComparison1.png)
![Comparison2](/Images/ClassifierComparison2.png)
![Comparison3](/Images/ClassifierComparison3.png)
![FeatureCost1](/Images/FeatureComputationCost.png)