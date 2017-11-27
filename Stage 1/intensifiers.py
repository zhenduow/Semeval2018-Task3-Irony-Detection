import nltk
import re
import nltk.data
from nltk import word_tokenize, pos_tag


# Idea of using intensifiers as a feature and list prepared by Madiha. Code adapted from Discourse Scoring by Kevin Swanberg.

def intensi_scorer(tweets):


    with open('intensifiers_list.txt', 'r') as file:
        intensifiers_list = file.read()
        intensifiers_list = intensifiers_list.split('\n')

    intent_score_list = []

    for tweet in tweets:
        tweet = tweet.split()
        count = 0
        intent_count = 0
        for word in tweet:
            if word in intensifiers_list:
                intent_count += 1
            else:
                intent_count = intent_count
            count += 1


        if intent_count == 0:
            intent_score = 0
        else:
            intent_score = (intent_count / count)

        intent_score_list.append(intent_score)

    return intent_score_list
