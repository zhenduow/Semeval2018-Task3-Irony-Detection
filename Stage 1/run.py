# -*- coding: utf-8 -*-

# Main function
# AUTHOR: Zhenduo Wang
# The main function calls all the feature measuring function to get the measurements
#		and then concatenate the measurements into feature matrix. Then it calls
#		the classifiers function to get the prediction label.

import numpy

from affectiveFeatures import *
from classifiers import *
from ij_score import interjection_score
from posfunctions import *
from preProcessing import *
from political import political_scorer
from sentencesimilarity import *
from structuralfeatures import *
from intensifiers import intensi_scorer

# ----------MAIN RUN FUNCTION--------#

with open('SemEval2018-T4-train-taskA.txt', encoding='utf-8') as f:
    tweet_list = [line.split('\t')[2] for line in f]

with open('SemEval2018-T4-train-taskA.txt', encoding='utf-8') as f:
    score_list = [line.split('\t')[1] for line in f]

# Removes label from from the tweet list and the score list
tweet_list = tweet_list[1:]
score_list = score_list[1:]

# Prepare an empty array
cleaned_score_list = []

# This loop removes newline characters from the score list and converts the scores from strings to float values
for item in score_list:
    intscore = item.rstrip()
    intscore = float(intscore)
    cleaned_score_list.append(intscore)

print("Cleaning tweets...")
# Call Preprocess on the Tweet Data
cleanedtweets = preprocess(tweet_list)

print("Computing polarity and subjectivity...")
# Create an array containing the polarity and subjectivity scores
polarity_and_subjectivity = PolarityAndSubjectivity(cleanedtweets)

print("Computing similarities...")
# Collect the sentence similarity of all the cleaned tweet data
sent_sim_list = simrun(cleanedtweets)

print("Computing discourse markers...")
# Collect the discourse marker scores of all the cleaned tweet data
disc_list = discourse_scorer(cleanedtweets)

print("Computing named entity count...")
# Collect the named entity scores of all the cleaned tweet data
ne_list = named_entity_count(cleanedtweets)

print("Computing intensifiers")
# Collect the intensifier scores of all the cleaned tweet data
inten_list = intensi_scorer(cleanedtweets)

print("Computing political")
# Collect the political scores of all the cleaned tweet data
pol_list = political_scorer(cleanedtweets)

print("Computing celebrity")
# Collect the celebrity mention scores of all the cleaned tweet data
celeb_list = celebrity_scorer(cleanedtweets)

print("Computing adjectives and adverbs...")
# Collect the adjective and adverb scores of all the cleaned tweet data
adj_adv_list = adj_adv_counter(cleanedtweets)

print("Computing punctuation markers...")
# Collect the punctuation marker scores of all the cleaned tweet data
punc_list = punc_count(tweet_list)

wc_list = word_counter(cleanedtweets)

print("Concatenating features...")
# Concatenate all the features together
feature_table = numpy.column_stack(
    [polarity_and_subjectivity, sent_sim_list, pol_list, disc_list, celeb_list, ne_list, inten_list, adj_adv_list, punc_list, wc_list])

# Using classifiers and generate output
print("Training Classifiers...")
X = numpy.array(feature_table)
y = numpy.array(cleaned_score_list)


output = open("output_Lovelace.txt", 'w')

LogisticRegressionClf(X, y, output)
SVMClf(X, y, output)
RandomForestClf(X, y, output)

output.close()
