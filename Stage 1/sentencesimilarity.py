# -*- coding: utf-8 -*-
import nltk.data
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn


#-------Sentence Similarity Functions---------#
#AUTHOR: Kevin Swanberg
# Adapted from: http://nlpforhackers.io/wordnet-sentence-similarity/

#Import in main run function as from sentencesimilarity import Sentencesimilarity
#Call in main run function as Sentencesimilarity.simrun(tweets)



#It was noted that ironic tweets often contain disparate semantic similarity between sentences - in other words,
#a tweet with two sentences is talking about two different things (probably comparing them to each other) in the two
#sentences, and this is often a feature of ironic tweets

#This set of functions computes the semantic similarity for tweets with two sentences. If they only have one sentence,
#obviously, there cannot be disparate semantic similarity between sentences since there is no "between sentences." If
#there are more than two sentences our code cannot currently handle this. At this time, in either of these scenarios
#(Only one sentence, or more than two sentences) the tweet is simply given a similarity score of 1. For stage 2,
#Semantic similarity will be calculated for tweets with two or more sentences as well.

#This module works by first tagging the parts of speech of each word in each tweet using nltk's tokenize feature. This
#set of tokenized words is then converted from Penn POS tags to Wordnet POS tags (reasoning explained in detail later).
#Next, these words and their POS tags are passed to wordnet's synset feature. where a list of their synonyms is generated.
#The synonym list for each word in each sentence is compared to the other sentence, and if a match is found, this increases
#The semantic similarity score.


def penn_to_wn(tag): #This code not written by Kevin - taken from the source cited above
	# This function converts Penn tags to Wordnet tags.
	# This is done because our code uses NLTK's POS tagging system, which uses the Penn POS Tags. However, Wordnet uses different
	# Tags and Wordnet is necessary for their synset feature. It's important to note that Wordnet ONLY has Synsets for Nouns, Verbs,
	# Adjectives, and Adverbs, so currently other POS's are ignored

	if tag.startswith('N'):
		return 'n'

	if tag.startswith('V'):
		return 'v'

	if tag.startswith('J'):
		return 'a'

	if tag.startswith('R'):
		return 'r'

	return None

def tagged_to_synset(word, tag): #This code not written by Kevin - taken from the source cited above
	# This function calls Wordnet synsets to get the semantic similarity scores for words.

	#Call the penn to wordnet tag conversion function for each tag
	wn_tag = penn_to_wn(tag)

	#if the tag is none(as in, not an adjective, adverb, verb, or noun) then return none - we cannot get synsets for other POS's
	if wn_tag is None:
		return None

	#Get the synset (synonyms) for each word in the sentence, return this
	try:
		return wn.synsets(word, wn_tag)[0]
	except:
		return None

def sentence_similarity(sentence1, sentence2): #This code adapted by Kevin Swanberg
	# This function computes the similarity between two sentences, one word at a time. First it tags each word, retrieves the synsets

	# Tag words in sentences
	sentence1 = pos_tag(word_tokenize(sentence1))
	sentence2 = pos_tag(word_tokenize(sentence2))

	# Retrieve synsets
	synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
	synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

	# Remove null values
	synsets1 = [ss for ss in synsets1 if ss]
	synsets2 = [ss for ss in synsets2 if ss]

	score, count = 0.0, 0

	# This for loop loops through the first sentence, gets the similarity score for each word between the two sentences
	#There is a nested for loop that loops through the words in the second sentence
	#I had to fix this loop - the original crashed every time.

	for syn1 in synsets1:

		#List of similarity scores for each pass through the sentence
		arr_simi_score = []
		simi_score = 0
		for syn2 in synsets2:

			simi_score = syn1.path_similarity(syn2)
		if simi_score is not None:
			arr_simi_score.append(simi_score)
		if (len(arr_simi_score) > 0):
			best = max(arr_simi_score) #Take the 'best' similarity score - the one the found the best match of synonyms
			score += best
			count += 1
	if score > 1: score = 1
	return score

#Main Sentence Similarity Function By Kevin Swanberg
def simrun(tweets):
	sent_sim_list = []

	#iterate through each tweet
	for tweet in tweets:
		sentences = (nltk.sent_tokenize(tweet)) #Split into sentences

		#Sentence similarity can only be calculated if there are two sentences in a tweet. If there are two sentences,
		#Calculated sentence similarity. If not, return a sentence similarity score of 1
		if len(sentences) > 1 and len(sentences[1]) > 3:
			focus_sentence = sentences[0]
			sentence = sentences[1]
			sent_sim_list.append(sentence_similarity(focus_sentence, sentence))
		else:
			sent_sim_list.append(1)

	#Return a list containing sentence similarity values
	return sent_sim_list
