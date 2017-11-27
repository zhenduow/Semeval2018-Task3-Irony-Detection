# -*- coding: utf-8 -*-
# affectiveFeatures
# AUTHOR:Zhenduo Wang
# This script measures fine-grained emotion information and sentiment polarity and subjectivity.
# Each feature has a function with explanation of how we measure and the reason to include that feature.

from textblob import TextBlob

"""This function measures of affective score of the emoticons and other emotional words.
    The sub-emotional information are proved to be very useful in finer-grained emotion classification.
	See Delia Irazú Hernańdez Farías, Viviana Patti, and Paolo Rosso. 2016. 
		Irony Detection in Twitter: The Role of Affective Content. ACM Trans. 
		Internet Technol. 16, 3, Article 19 (July 2016), 24 pages. DOI: http://dx.doi.org/10.1145/2930663
	More information about the Emolexicon: http://saifmohammad.com/WebPages/lexicons.html
    """
def EmoticonVector(tweet_list):
	emoticonVectorList = []
	for i,tweet in enumerate(tweet_list):
		#initiate emoticon vector
		emoticonVector = numpy.array([0,0,0,0,0,0])
		tokenList = tweet.split()
		#for each word, look up in the lexicon for sub-emotional features vector
		for emoticon in tokenList:
			emotivec = []
			wb = openpyxl.load_workbook('emosenticnet.xlsx')
			sheet = wb.get_sheet_by_name('EmoSenticNet')
			for row in sheet.iter_rows():
				for cell in row:
					if cell.value == emoticon:
						rowIndex = cell.row
						if rowIndex > 1:
							emotivec.append(int(sheet['B' + str(rowIndex)].value))
							emotivec.append(int(sheet['C' + str(rowIndex)].value))
							emotivec.append(int(sheet['D' + str(rowIndex)].value))
							emotivec.append(int(sheet['E' + str(rowIndex)].value))
							emotivec.append(int(sheet['F' + str(rowIndex)].value))
							emotivec.append(int(sheet['G' + str(rowIndex)].value))
							emotivec = numpy.array(emotivec)
							emoticonVector += emotivec
		emoticonVectorList.append(emoticonVector.tolist())
	return emoticonVectorList
	

"""This function measures the sentiment polarity and subjectivity using the TextBlob package.
    Previous work proved sentiment polarity and subjectivity is correlated with irony in tweets.
	Irony irony tend to have positive sentiment words and be more subjective.
	See Delia Irazú Hernańdez Farías, Viviana Patti, and Paolo Rosso. 2016. 
		Irony Detection in Twitter: The Role of Affective Content. ACM Trans. 
		Internet Technol. 16, 3, Article 19 (July 2016), 24 pages. DOI: http://dx.doi.org/10.1145/2930663
    """
def PolarityAndSubjectivity(tweet_list):
	measurementsList = []
	for tweet in tweet_list: 
		testimonial = TextBlob(tweet)
		#The polarity score is a float within the range [-1.0, 1.0]. 
		#The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective
		measurementsList.append([testimonial.sentiment.polarity, testimonial.sentiment.subjectivity])
	return measurementsList