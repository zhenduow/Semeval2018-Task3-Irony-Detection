#!/usr/bin/env python

"""
evaluate.py
This is the scoring script for SemEval-2018 Task 3: Irony detection in English tweets.
The script:
  * is used to evaluate Task A and Task B
  * takes as input a submission dir containing the system output (format: 1 prediction per line)
  * prediction files should be named 'predictions-taskA.txt'
  * calculates accuracy, precision, recall and F1-score.
Date: 10.13.2017
"""

def precision_recall_fscore(true, predicted, beta=1, labels=None, pos_label=None, average=None):
	"""Calculates the precision, recall and F-score of a classifier.
	:param true: iterable of the true class labels
	:param predicted: iterable of the predicted labels
	:param beta: the beta value for F-score calculation
	:param labels: iterable containing the possible class labels
	:param pos_label: the positive label (i.e. 1 label for binary classification)
	:param average: selects weighted, micro- or macro-averaged F-score
	"""

	# Build contingency table as ldict
	ldict = {}
	for l in labels:
		ldict[l] = {"tp": 0., "fp": 0., "fn": 0., "support": 0.}

	for t, p in zip(true, predicted):
		if t == p:
			ldict[t]["tp"] += 1
		else:
			ldict[t]["fn"] += 1
			ldict[p]["fp"] += 1
		ldict[t]["support"] += 1

	# Calculate precision, recall and F-beta score per class
	beta2 = beta ** 2
	for l, d in ldict.items():
		try:
			ldict[l]["precision"] = d["tp"] / (d["tp"] + d["fp"])
		except ZeroDivisionError:
			ldict[l]["precision"] = 0.0
		try:
			ldict[l]["recall"] = d["tp"] / (d["tp"] + d["fn"])
		except ZeroDivisionError:
			ldict[l]["recall"] = 0.0
		try:
			ldict[l]["fscore"] = (1 + beta2) * (ldict[l]["precision"] * ldict[l]["recall"]) / (
			beta2 * ldict[l]["precision"] + ldict[l]["recall"])
		except ZeroDivisionError:
			ldict[l]["fscore"] = 0.0

	# If there is only 1 label of interest, return the scores. No averaging needs to be done.
	if pos_label:
		d = ldict[pos_label]
		return (d["precision"], d["recall"], d["fscore"])
	# If there are multiple labels of interest, macro-average scores.
	else:
		for label in ldict.keys():
			avg_precision = sum(l["precision"] for l in ldict.values()) / len(ldict)
			avg_recall = sum(l["recall"] for l in ldict.values()) / len(ldict)
			avg_fscore = sum(l["fscore"] for l in ldict.values()) / len(ldict)
		return (avg_precision, avg_recall, avg_fscore)


# Get feature significance - From website https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
def show_most_informative_features(clf, n, output):
	feature_names = ["polarity",
	                 "subjectivity",
	                 "similarity",
	                 "discourse marker",
	                 "named entity",
	                 "adjective/adverb",
	                 "punctuation",
	                 "word count",
	                 "celebrity",
	                 "political",
	                 "laughter",
	                 "preposition",
	                 "intensifiers",
	                 "stopwords"
	                 ]
	coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
	top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
	output.write("Feature weights:\n")
	for (coef_1, fn_1), (coef_2, fn_2) in top:
		output.write("\t%.4f\t%-15s\n" % (coef_1, fn_1))
