import nltk
import re
import nltk.data
from nltk import word_tokenize, pos_tag


# Idea of scoring political hashtags and list prepared by Madiha. Code adapted from Discourse Scoring by Kevin Swanberg.

def political_scorer(tweets):


    with open('political_list.txt', 'r') as file:
        political_list = file.read()
        political_list = political_list.split('\n')

        political_score_list = []

    for tweet in tweets:
        tweet = tweet.split()
        count = 0
        political_count = 0
        for word in tweet:
            if word in political_list:
                political_count += 1
            else:
                political_count = political_count
            count += 1


        if political_count == 0:
            political_score = 0
        else:
            political_score = (political_count / count)

        political_score_list.append(political_score)

    return political_score_list
