#!python2
###############################################################################
# MachineLearningExperiment.py - Machine Learning Comic Book Cover Finding Experiment
# jamesj223

###############################################################################
# Initialisation

from datetime import datetime

import cPickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, confusion_matrix

# ML Algorithms
from sklearn.dummy import DummyClassifier # Lol
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#from tqdm import tqdm

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
warnings.simplefilter(action='ignore', category=FutureWarning)

###############################################################################
# Config

###############################################################################
# Classes

###############################################################################
# Functions

def save_obj(obj, name, protocol=2):
  with open('obj/'+ name + '.pkl', 'wb') as f:
      cPickle.dump(obj, f, protocol)

###############################################################################
# Main

if __name__ == '__main__':   

    startTime = datetime.now()

    print("Start - " + str(datetime.now()))
    print("")

    ### Code Goes Here

    # Pandas/Train Stuff

    train = pd.read_csv('TrainingSet2.csv')

    y=train['label'].values

    x = train.drop('label', 1)

    # :( Need to figure out how to do this better
    # File names are kinda important for what I want...
    x = x.drop('fileName', 1) 

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

    # SK Learn Pipe Stuff

    # pipe = make_pipeline(
    #   StandardScaler(),
    #   #RandomForestClassifier(random_state=0)
    #   DummyClassifier(strategy="uniform")
    # )

    # pipe.fit(x_train, y_train)

    # accuracyScore = accuracy_score(pipe.predict(x_test), y_test)

    #print("Accuracy: {:.4%}".format(accuracyScore))

    # Multi Classifier Showdown
    classifiers = [
        KNeighborsClassifier(3),
        #SVC(kernel="linear", C=0.1, probability=True), 
        #NuSVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        #QuadraticDiscriminantAnalysis()
    ]

    for clf in classifiers:

        name = clf.__class__.__name__

        #pipe = make_pipeline(
        #   StandardScaler(),
        #   clf
        #)

        #pipe.fit(x_train, y_train)
        clf.fit(x_train, y_train)

        save_obj(clf, name)

        train_predictions = clf.predict(x_test)
        train_predictions_proba = clf.predict_proba(x_test)

        print("="*30)
        print(name)
        #print('****Results****')

        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))

        precision = precision_score(y_test, train_predictions)
        print("Precision: {}".format(precision))

        recall = recall_score(y_test, train_predictions)
        print("Recall: {}".format(recall))

        f1 = f1_score(y_test, train_predictions)
        print("F1 Score: {}".format(f1))

        logloss = log_loss(y_test, train_predictions_proba)
        print("Log Loss: {}".format(logloss))

        print("")
        C = confusion_matrix(y_test, train_predictions)
        print("True Positives: {}".format(C[1,1]))
        print("True Neagtives: {}".format(C[0,0]))
        print("False Positives: {}".format(C[0,1]))
        print("False Negatives: {}".format(C[1,0]))



        print("")


    ### Code Ends Here

    print("End - " + str(datetime.now()))

    print("Took: " + str( datetime.now() - startTime ))
