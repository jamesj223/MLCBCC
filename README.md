# Machine Learning Comic Book Cover Classifier

## Overview

A series of scripts to use machine learning to find and extract covers from comic books. 

Most comic book files will have the cover as the first page, but often they will have multiple covers. Sometimes these are included at the start, sometimes at the end, and sometimes they are spread throughout the book. The goal of this project is to be able to run it against a directory full of comic book files, and have it extract all of the covers, so that they can be used to generate a cool collage (see the examples section below)

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

MLE_2_Training_Attempt_1.py is the second main file. Given a training data set, split it 80:20 training:test, then run various different classifiers using those two sets and measure their performance.

Key metrics we are measureing are (Accuracy)[https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score], (Precision)[https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html], (Recall)[https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html], (F1)[https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html] and (Logistic Loss)[https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss].

The classifiers tested are:

- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
- [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier)
- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
- [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_-bayes.GaussianNB)
- [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

The results of the tests looked like this:

![Comparison3](/Images/ClassifierComparison3.png)

Overall, GradientBoostingClassifier was found to be the best option for this use case.

## Testing

## Early Experiments/Testing

## Misc Benchmarking

Comparison of different classifiers and how they performed against various training sets

![Comparison1](/Images/ClassifierComparison1.png)
![Comparison2](/Images/ClassifierComparison2.png)
![Comparison3](/Images/ClassifierComparison3.png)

Examining computation/time cost of different feature types

![FeatureCost1](/Images/FeatureComputationCost.png)