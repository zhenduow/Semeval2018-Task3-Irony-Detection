# -*- coding: utf-8 -*-
# Classifiers 
# AUTHOR:Zhenduo Wang
# This script implements 3 classifiers(Logistic Regression,SVM and Random Forest)
# All of them takes the feature matrix and produce the prediction label
# We keep tracking of these classifiers with different features in order to get the best results.

import numpy
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
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


def SVMClf1(X, y, index):
	index=int(index)
	clf = SVC()
	Xtest=X[index:index+1]
	ytest=y[index:index+1]
	Xtrain=X[:index]
	ytrain=y[:index]
	# Call the evaluation function
	clf.fit(Xtrain, ytrain)
	print(clf.predict(Xtest))


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
