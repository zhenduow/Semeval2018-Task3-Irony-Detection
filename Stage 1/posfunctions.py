# -*- coding: utf-8 -*-
import nltk
import nltk.data
from nltk import word_tokenize, pos_tag


#----------------PART OF SPEECH FUNCTIONS---------------#
#AUTHOR: Kevin Swanberg

#Import in main run file as from posfunctions import Posfunctions

#This file contains functions that require Part of Speech tagging, including Named Entity Detection and Adjective and
#Adverb Counting


#--------NAMED ENTITY DETECTION-------# By Kevin Swanberg
#This module identifies and counts 'named entities' in tweets. Named entities are 'proper nouns' such as names of people,
#places, organizations, businesses, or events. This takes advantage of the nltk tagset, as nltk already tags named entities
#as a POS, so it is very simple to simply search for the NE POS tag. Named entities were found to be more common in ironic
#tweets by researchers, motivating the use of this feature.

#Call in main run function as Posfunctions.named_entity_count(sample)

#This function tokenizes each tweet by separating it into sentences, then chunks sentences and creates an NLTK POS tree
#for each sentence
def named_entity_preprocess(data): #By Kevin Swanberg

	#Use nltk tokenizer to split given tweet into sentences
	sentences = nltk.sent_tokenize(data)

	#Split sentences into words, store these as lists of tuples [WORD, POS], then Mark POS of each word in each sentence
	tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
	tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
	chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
	return chunked_sentences

#This function identifies if a feature is 'NE (Named Entity) in the list of tuples
def extract_entity_names(t): #By Kevin Swanberg
	entity_names = []

	#If the word has the attribute 'label' (marks POS)
	if hasattr(t, 'label') and t.label():

		#If the label is 'NE' (Named Entity)
		if t.label() == 'NE':

			#Make a list of the named entities
			entity_names.append(' '.join([child[0] for child in t]))
		else:
			for child in t:
				entity_names.extend(extract_entity_names(child))

	return entity_names

#This is the main named entity count function, it iterates through a list of tweets, chunks each tweet and counts
#the named entities in each tweet. It then divides the number of named entities by the total word count for the tweet
#in order to normalize the number for the tweet's size.
def named_entity_count(sample): #By Kevin Swanberg

	#Make a list of the number of named entities in each tweet
	named_entity_list = []

	#Go through each tweet in the data set
	for tweet in sample:

		#Pass the tweet to the tagger fucntion
		chunked_sentences = named_entity_preprocess(tweet)

		#Make a list for the total number of named entities in each tweet
		entity_names = []

		#Iterates through each tree (sentence) in a tweet - since a tweet may have multiple sentences
		for tree in chunked_sentences:

			#Extract all the named entities and store them in the list for each sentence
			entity_names.extend(extract_entity_names(tree))

		#Take the length of the list of named entities - this is the total number of named entities in a single tweet
		ne_count = len(entity_names)

		#Take the total words in a tweet
		word_count = len(tweet.split())

		#Normalize the total named entity count for each tweet by dividing it by the word count of each tweet
		ne_score = ne_count / word_count

		#Add this 'score' to a list for all the tweets
		named_entity_list.append(ne_score)

	#Return a list of the 'named entity scores' for all tweets
	return named_entity_list

#-------ADJECTIVE AND ADVERB COUNTER-------# By Kevin Swanberg
#This module counts the total number of adjectives and adverbs and passes these counts as a list of tuples containing the total
#number of adjectives and the number of adverbs separately. Several researchers cited the frequency of adjectives and adverbs
#could be used to identify irony, however these researchers also counted the number of nouns and verbs. When we counted
#the nouns and verbs it reduced our accuracy, so we did not include these counts. The function works by using NLTK's word
#tokenizer to tag the POS of each word. These are then simply checked for the three adjective tags, 'JJ,' 'JJR,', and 'JJS'
#and the three adverb tags, 'RB,' 'RBS,' and 'RBR.' The 'score' of the adjective and adverb counts are normalized by
#dividing the total count for each tweet by the word count for each tweet in order to account for length.

#Call in main run function as Posfunctions.adj_adv_counter(tweets)

def adj_adv_counter(tweets): #By Kevin Swanberg

	adj_adv_list = []

	#Iterate through tweets
	for tweet in tweets:

		#Zero all counters and tag a tokenized version of each tweet
		tagged = pos_tag(word_tokenize(tweet))
		word_count = 0
		adj_count = 0
		adv_count = 0
		adj_score = 0
		adv_score = 0
		feature_list = []

		#Iterate through each tagged word
		for tag in tagged:

			#If the word is tagged as an adjective, add one to the count of adjectives
			if (tag[1] == 'JJ') or (tag[1] == 'JJR') or (tag[1] == 'JJS'):
				adj_count += 1

			#If the word is tagged as an adverb, add one to the count of adverbs
			if (tag[1] == 'RB') or (tag[1] == 'RBS') or (tag[1] == 'RBR'):
				adv_count += 1

			#keep track of word count
			word_count += 1

		#Divide adjective and adverb counts by total word count to normalize for tweet length
		adj_score = adj_count / word_count
		adv_score = adv_count / word_count

		#Output a list of the adjective and adverb scores for each tweet
		feature_list = [adj_score, adv_score]
		adj_adv_list.append(feature_list)

	#return the list of the adjective and adverb counts for every tweet
	return adj_adv_list


