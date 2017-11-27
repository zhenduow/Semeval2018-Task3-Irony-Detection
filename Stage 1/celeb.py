import nltk
import re
import nltk.data
from nltk import word_tokenize, pos_tag


# Idea of scoring celebrity hashtags and list prepared by Madiha. Code adapted from Discourse Scoring by Kevin Swanberg.
#List includes entertainment, media, music, reality tv, sports, fashion

def celebrity_scorer(tweets):


    with open('celebrity_list', 'r') as file:
        celebrity_list = file.read()
        celebrity_list = celebrity_list.split('\n')

        celebrity_score_list = []

    for tweet in tweets:
        tweet = tweet.split()
        count = 0
        celebrity_count = 0
        for word in tweet:
            if word in celebrity_list:
                celebrity_count += 1
            else:
                celebrity_count = celebrity_count
            count += 1


        if celebrity_count == 0:
            celebrity_score = 0
        else:
            celebrity_score = (celebrity_count / count)

        celebrity_score_list.append(celebrity_score)

    return celebrity_score_list
