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
from sklearn.naive_bayes import GaussianNB
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

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

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
	print_cm(confusion_matrix(y[train_size:],clf.predict(X[train_size:])),labels=['non-ironic','ironic'])
		
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
	#output.write("predicted\tnon-ironic\tironic\n")
	#output.write("true\n")
	#output.write("non-ironic")
	#output.write(confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist()[0,0])
	#output.write(confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist()[0,1])
	#output.write("ironic")
	#print (confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist())
	#output.write(confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist()[1,0])
	#output.write(confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist()[1,1])
	for item in confusion_matrix(y[train_size:],clf.predict(X[train_size:])).tolist():
		output.write("%s\n" % item)
	print_cm(confusion_matrix(y[train_size:],clf.predict(X[train_size:])),labels=['non-ironic','ironic'])
	
	
def NBClf(X, y, output):
	sample_size = len(X)
	train_size = int(0.8 * sample_size)
	feature_num = len(X[0])
	folds = 5
	clf = GaussianNB()
	# Cross validation accuracy  
	scores = cross_val_score(clf, X, y, cv=folds)
	# Call the evaluation function
	clf.fit(X[:train_size], y[:train_size]) 
	p, r, f = precision_recall_fscore(y[train_size:], clf.predict(X[train_size:]), beta=1, labels=[0,1], pos_label=1)
	output.write("Naive Bayes Classifier:\n")
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
	print_cm(confusion_matrix(y[train_size:],clf.predict(X[train_size:])),labels=['non-ironic','ironic'])


def VotedClf(X,y,output):
	sample_size = len(X)
	train_size = int(0.8 * sample_size)
	feature_num = len(X[0])
	folds = 5
	clf1 = SVC()
	clf2 = LogisticRegression()
	clf3 = RandomForestClassifier()
	vclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
	vclf.fit(X, y)
	# Cross validation accuracy
	scores = cross_val_score(vclf, X, y, cv=folds)
	# Call the evaluation function
	vclf.fit(X[:train_size], y[:train_size])
	p, r, f = precision_recall_fscore(y[train_size:], vclf.predict(X[train_size:]), beta=1, labels=[0, 1], pos_label=1)
	output.write("Voted Classifier:\n")
	output.write('Mean Accuracy:\n')
	output.write(str(numpy.mean(scores)))
	output.write("\np=:")
	output.write(str(p))
	output.write("\nr=:")
	output.write(str(r))
	output.write("\nf=:")
	output.write(str(f))
	output.write("\nConfusion Matrix:\n")
	for item in confusion_matrix(y[train_size:], vclf.predict(X[train_size:])).tolist():
		output.write("%s\n" % item)


def bowpca(X):
	pca = PCA(n_components=30)
	pca.fit(X)
	return X
