# -*- coding: utf-8 -*-

#-------------STRUCTURAL FEATURES---------#
#AUTHOR: Kevin Swanberg
#This file contains simple structural feature measures including discourse markers, punctuation, and word count

#Import in main run file as from structuralfeatures import Structuralfeatures

#--------DISCOURSE SCORING--------# By Kevin Swanberg
#This set of functions counts the total number of 'Discourse Markers' in each tweet and normalizes this by the word count
#of each tweet. Discourse markers are claimed to appear more often in tweets containing irony and sarcasm. This module works
#by checking each word in a tweet against a manually created list of known discourse markers. These were found by comparing
#lists of discourse markers online and selecting the most common ones. There are 53 discourse markers in the list -
#extended from the Stage 1 submission

#In addition to Discourse markers, we also check for "Laughter" markers, Swear Words, and Stopwords - while not technically discourse
#Markers, these are similar in their use and in their identification, so they are included in this section.

#Call in main run function as Structuralfeatures.discourse_scorer(tweets)

import re

def discourse_scorer(tweets): #By Kevin Swanberg

	#Open the pre-made list of discourse markers
	with open('discourse_list.txt', 'r') as file:
		discourse_list = file.read()
		discourse_list = discourse_list.split('\n')

	disc_score_list = []

	#Check each word in each tweet for words from the discourse marker list. If there is a discourse marker, increase
	#the discourse count by one
	for tweet in tweets:
		tweet = tweet.split()
		count = 0
		disc_count = 0
		for word in tweet:
			if word in discourse_list:
				disc_count += 1
			else:
				disc_count = disc_count
			count +=1

		#The final Discourse score is the number of discourse markers divided by the total number of words in each tweet
		if disc_count == 0:
			disc_score = 0
		else:
			disc_score = (disc_count / count)

		disc_score_list.append(disc_score)

	return disc_score_list

def laughter_scorer(tweets): #By Kevin Swanberg
	#Similar to the discourse markers, this was a feature implemented without any academic research behind it, but the
	#theory was that these would be prevalanet in ironic tweets, just like discourse markers. Since they were not
	#*technically* laughter markers are not discourse markers. The function works by checking if "laughter" markers are
	#contained in each tweet, adds one for every time one of these occurs, then normalizes these occurances for the length
	#of the tweet using the word count
	#initialize the list
	laugh_score_list = []
	#iterate through each tweet
	for tweet in tweets:
		#split tweet into words
		tweet = tweet.split()
		#initialize word count and laugh count
		count = 0
		laugh_count = 0
		#iterate through words
		for word in tweet:
			#check for laughter markers, add one each time one occurs
			if word in ("lol", "rofl", "haha", "ha", "hah", "lmfao", "lmao", "lmaoo", "lmfaoo", "hahaha"):
				laugh_count += 1
			count+=1
		#normalize for length of tweet
		laugh_score = (laugh_count / count)
		#append the tweet's score to the overall list and return the list
		laugh_score_list.append(laugh_score)
	return laugh_score_list

def swear_scorer(tweets): #By Kevin Swanberg
	#Again, there is no academic basis behind this feature, but since ironic tweets are typically emotional in nature
	#it was hypothesized that ironic tweets might include these "emotional words."The function works by checking if swear
	#words are contained in each tweet, adds one for every time one of these occurs, then normalizes these occurances for
	#the length of the tweet using the word count

	#Open the pre-made list of swear words
	with open('swear_list.txt', 'r') as file:
		swearwords = file.read()
		swearwords = swearwords.split('\n')
	#initalize the list
	swear_list = []
	#iterate through each tweet
	for tweet in tweets:
		#split tweet into words
		tweet = tweet.split()
		#initialize word count and swear count
		wc = 0
		swearcount = 0
		#iterate through each word
		for word in tweet:
			#if the word is a swear word, add one to the swear count
			if word in swearwords:
				swearcount +=1
			wc+=1
		#normalize for length of tweet and add score to the swear list
		swearscore = swearcount/wc

		swear_list.append(swearscore)

	return swear_list

def stopwords_score(tweets): #By Kevin Swanberg

	#Stopwords are words that search engines and NLP programs often ignore because they do not contain significant information
	#often. However, they are very common in conversational English, which is a common feature of irony and so we thought
	#this may be a significant feature - it did offer some improvement for our system


	#Open the pre-made list of stopwords markers
	with open('stopwords.txt', 'r') as file:
		stopwords = file.read()
		stopwords = stopwords.split('\n')
	#initialize list
	stop_list = []
	#iterate through tweets
	for tweet in tweets:
		#split tweet into words
		tweet = tweet.split()
		#initilaize counters
		wc = 0
		stopcount = 0
		#iterate through words
		for word in tweet:
			#if the word is a stopword, add to the stopword count, add to wordcount
			if word in stopwords:
				stopcount +=1
			wc+=1
		#normalize for length of tweet
		stopscore = stopcount/wc
		#append to list
		stop_list.append(stopscore)
	#return the list
	return stop_list

#--------PUNCTUATION COUNTER------# By Kevin Swanberg
#This function counts punctuation deemed significant by researchers on automatic irony detection. These
#punctuation types are said to occur in tweets containing irony typically. First, the function counts each punctuation
#type seperately using simple regex, then passes each of these counts as a list of features to a master list containing
#the feature counts for each tweet.

#Call in main run function as Structuralfeatures.punc_count(tweets)

def punc_count(tweets): #By Kevin Swanberg

	punc_count_list = []

	for tweet in tweets:

		#Zero all variables for each tweet
		feature_list = []
		hash_count = 0
		ellipsis1_count = 0
		ellipsis2_count = 0
		exclamation_count = 0
		question_count = 0
		colon_count = 0
		quote_count = 0
		apostrophe_count = 0

		#Check for each important piece of punctuation
		hash_count = tweet.count('#')
		ellipsis1_count = (tweet.count('..') - tweet.count('...'))
		ellipsis2_count = tweet.count('...')
		exclamation_count = tweet.count('!')
		question_count = tweet.count('?')
		colon_count = tweet.count(':')
		quote_count = tweet.count('"')
		apostrophe_count = tweet.count("'")

		#Make a list of the count of each piece of punctuation and append this to the count of punctuation in all tweets
		feature_list = [hash_count, ellipsis1_count, ellipsis2_count, exclamation_count, question_count, colon_count, quote_count, apostrophe_count]
		punc_count_list.append(feature_list)

	#return a list containing the individual punctuation counts for every tweet
	return punc_count_list

#-------WORD COUNTER-------# By Kevin Swanberg
#It was claimed by researchers that word count was a significant indicator of tweets containing irony, with irony being associated
#with tweets with lower word counts. The function simply splits a tweet into a list of words, then takes the length of this
#list as the word count, and returns a list of these word counts. For stage 2, we tried counting the word count of each
#individual sentence within each tweet, but this actually reduced our accuracy, so we kept this simple by just taking the
#word count of each tweet.

#Call in main run function as Structuralfeatures.word_counter(tweets)

def word_counter(tweets): #By Kevin Swanberg

	word_count_list = []

	#iterate through each tweet
	for tweet in tweets:

		#split the tweet into words, take the length of this list
		wordcount = len(tweet.split())

		#append it to the list of wordcounts for every tweet
		word_count_list.append(wordcount)

	#return the list of word counts
	return word_count_list

#---URL COUNTER----#
#This function idnetifies if a tweet has a URL. This was designed based on the work found in "Detecting Sarcasm in
# Multimodal Social Platforms" (DOI 10.1145/2964284.2964321) - They found that ironic tweets often contain images
#and often the irony depends on the image. However, our data does not immediately give us images. We did see when
#assessing the data though that any time there was an image, it was included in the link using a URL, and a majority
# of the URLs in the data were images, so the simplest way to identify this was to just check if a tweet had a URL

def url_count(tweets): #By Kevin Swanberg
	#initialize the list
	url_list = []
	#iterate through tweets
	for tweet in tweets:
		#regex for URLs
		if (re.search("(?P<url>https?://[^\s]+)", tweet)) is not None:
			url_list.append(1)
		else:
			url_list.append(0)
	return url_list