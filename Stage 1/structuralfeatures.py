# -*- coding: utf-8 -*-

#-------------STRUCTURAL FEATURES---------#
#AUTHOR: Kevin Swanberg
#This file contains simple structural feature measures including discourse markers, punctuation, and word count

#Import in main run file as from structuralfeatures import Structuralfeatures

#--------DISCOURSE SCORING--------# By Kevin Swanberg
#This set of functions counts the total number of 'Discourse Markers' in each tweet and normalizes this by the word count
#of each tweet. Discourse markers are claimed to appear more often in tweets containing irony and sarcasm. This module works
#by checking each word in a tweet against a manually created list of known discourse markers. These were found by comparing
#lists of discourse markers online and selecting the most common ones. Currently there are only 33 discourse markers but
#more will be added for the stage 2 submission.

#Call in main run function as Structuralfeatures.discourse_scorer(tweets)

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

def laughter_scorer(tweets):
	laugh_score_list = []
	for tweet in tweets:
		tweet = tweet.split()
		count = 0
		laugh_count = 0
		for word in tweet:
			if word in ("lol", "rofl", "haha", "ha", "hah", "lmfao", "lmao", "lmaoo", "lmfaoo", "hahaha"):
				laugh_count += 1
			else:
				laugh_count = laugh_count
			count+=1

		if laugh_count == 0:
			laugh_score = 0
		else:
			laugh_score = (laugh_count / count)
		laugh_score_list.append(laugh_score)
	return laugh_score_list

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
#list as the word count, and returns a list of these word counts

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

def stopwords_score(tweets):

	#Open the pre-made list of discourse markers
	with open('stopwords.txt', 'r') as file:
		stopwords = file.read()
		stopwords = stopwords.split('\n')

	stop_list = []
	for tweet in tweets:
		tweet = tweet.split()
		wc = 0
		stopcount = 0
		for word in tweet:

			if word in stopwords:

				stopcount +=1
			wc+=1

		stopscore = stopcount/wc

		stop_list.append(stopscore)

	return stop_list