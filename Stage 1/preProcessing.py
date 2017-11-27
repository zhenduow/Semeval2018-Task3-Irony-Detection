# -*- coding: utf-8 -*-

"""
preprocessing.py
Code Author: Madiha Mirza
Tweets are unstructured data and contain many elements that have no syntactic function. This script cleans tweets 
to remove quotation marks, contractions, Twitter @usernames (@-mentions), Twitter #hashtags, URLs, digits, emoticons, 
transport & map symbols, pictographs, flags (iOS), and then performs case normalization and tokenization. 
Date: 10.13.2017
"""

import nltk
import re
import nltk.data
from nltk import word_tokenize, pos_tag


#This function cleans tweet data for running through the semantic similarity scoring and discourse marker scoring
def preprocess(tweets):

	cleanedtweets = []
	for tweet in tweets:

		# Removing hashtags
		tweet = re.sub(r'#', '', tweet)

		# Removing 0 and replacing with o
		tweet = re.sub(r'0', 'o', tweet)

		# Removing contractions
		tweet = re.sub(r'n\'t', r' not', tweet)

		# Removing Links
		tweet = re.sub(r"http\S+", "", tweet)

		# Removing Quotation
		tweet = re.sub("\"", "", tweet)

		#Make Lowercase
		tweet = tweet.lower()

		#Remove Emojis
		#UNICODE range are in narrow build now, and it remains the same functionality for wide build.
		emoji_pattern = re.compile("["
		                           u"\U0001F600-\U0001F64F"  # emoticons
		                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
		                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
		                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
		                           "]+", flags=re.UNICODE)
		tweet = emoji_pattern.sub(r'', tweet) # no emoji

		cleanedtweets.append(tweet)
	return cleanedtweets
