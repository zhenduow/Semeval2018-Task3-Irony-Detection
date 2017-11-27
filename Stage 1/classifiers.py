# -*- coding: utf-8 -*-
# Classifiers 
# AUTHOR:Zhenduo Wang
# This script implements 3 classifiers(Logistic Regression,SVM and Random Forest)
# All of them takes the feature matrix and produce the prediction label
# We keep tracking of these classifiers with different features in order to get the best results.

import numpy
from sklearn import *
from sklearn.linear_model import LogisticRegression  
from sklearn.svm import SVC
from textblob import TextBlob
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from evaluate import *

'''
---------- Support Vector Machine Classifier ------------
This function uses SVM classifier for the binary classification problem based on the feature we select. 
The classifier takes feature matrix as input and outputs the categorical label vector (other classifiers as well).
We implent the classifier with sklearn package. 
We use cross validation to get accuracy of the classifier. 
We calculate precision, recall and F-measure score with the official evaluation function. 
We also compute confusion matrix with sklearn. 
'''
def SVMClf(X, y, output):
	sample_size = len(X)
	train_size = int(0.8 * sample_size)
	feature_num = len(X[0])
	folds = 5
	clf = SVC()
	# Cross validation accuracy  
	scores = cross_val_score(clf, X, y, cv=folds)
	# Call the evaluation function
	clf.fit(X[:train_size], y[:train_size]) 
	p, r, f = precision_recall_fscore(y[train_size:], clf.predict(X[train_size:]), beta=1, labels=[0,1], pos_label=1)
	output.write("Support Vector Machine Classifier:\n")
	output.write('Mean Accuracy:\n')
	output.write(str(numpy.mean(scores)))
	output.write("\np=:") 
	output.write(str(p))
	output.write("\nr=:") 
	output.write(str(r))
	output.write("\nf=:") 
	output.write(str(f))
	output.write("\nConfusion Matrix:\n")
	for item in confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist():
		output.write("%s\n" % item)
		
''' Logistic Regression Classifier
Because logistic regression has linear kernel, 
we get the weight for all the features with a third party function show_most_informative_features.
The magnitude of a weight implies the importance of that feature.
'''
def LogisticRegressionClf(X, y, output):
	sample_size = len(X)
	train_size = int(0.8 * sample_size)
	feature_num = len(X[0])
	folds = 5
	clf = LogisticRegression()
	# Cross validation accuracy  
	scores = cross_val_score(clf, X, y, cv=folds)
	# Call the evaluation function
	clf.fit(X[:train_size], y[:train_size]) 
	p, r, f = precision_recall_fscore(y[train_size:], clf.predict(X[train_size:]), beta=1, labels=[0,1], pos_label=1)
	output.write("Logistic Regression Classifier:\n")
	output.write('Mean Accuracy:\n')
	output.write(str(numpy.mean(scores)))
	output.write("\np=:") 
	output.write(str(p))
	output.write("\nr=:") 
	output.write(str(r))
	output.write("\nf=:") 
	output.write(str(f))
	output.write("\nConfusion Matrix:\n")
	for item in confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist():
		output.write("%s\n" % item)

	# Call the feature importance function
	show_most_informative_features(clf,feature_num,output)
	
# Random Forest Classifier 
def RandomForestClf(X, y, output):
	sample_size = len(X)
	train_size = int(0.8 * sample_size)
	feature_num = len(X[0])
	folds = 5
	clf = RandomForestClassifier()
	# Cross validation accuracy  
	scores = cross_val_score(clf, X, y, cv=folds)
	# Call the evaluation function
	clf.fit(X[:train_size], y[:train_size]) 
	p, r, f = precision_recall_fscore(y[train_size:], clf.predict(X[train_size:]), beta=1, labels=[0,1], pos_label=1)
	output.write("Random Forest Classifier:\n")
	output.write('Mean Accuracy:\n')
	output.write(str(numpy.mean(scores)))
	output.write("\np=:") 
	output.write(str(p))
	output.write("\nr=:") 
	output.write(str(r))
	output.write("\nf=:") 
	output.write(str(f))
	output.write("\nConfusion Matrix:\n")
	for item in confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist():
		output.write("%s\n" % item)
