#!/usr/bin/env python
# -*- coding: utf-8 -*-
#from sklearn.datasets import load_files


import nltk
#text = "Hey goody Alice and Ram, we are going outside. Would you like to come with us or sit with smart John ? Anyways the decision is yours. Your level of understanding is higher than us!!! We will come back very quick. Goodbye ! See you later."

text = "Ad sales boost Time Warner profit.  Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn (£600m)  for the three months to December, from $639m year-earlier. The firm, which is now one of the biggest investors in Google, benefited from sales of high-speed internet connections and higher advert sales. TimeWarner said fourth quarter sales rose 2% to $11.1bn from $10.9bn. Its profits were buoyed by one-off gains which offset a profit dip at Warner Bros, and less users for AOL. Time Warner said on Friday that it now owns 8% of search-engine Google. But its own internet business, AOL, had has mixed fortunes. It lost 464,000 subscribers in the fourth quarter profits were lower than in the preceding three quarters. However, the company said AOL's underlying profit before exceptional items rose 8% on the back of stronger internet advertising revenues. It hopes to increase subscribers by offering the online service free to TimeWarner internet customers and will try to sign up AOL's existing customers for high-speed broadband. TimeWarner also has to restate 2000 and 2003 results following a probe by the US Securities Exchange Commission (SEC), which is close to concluding. Time Warner\'s fourth quarter profits were slightly better than analysts' expectations. But its film division saw profits slump 27% to $284m, helped by box-office flops Alexander and Catwoman, a sharp contrast to year-earlier, when the third and final film in the Lord of the Rings trilogy boosted results. For the full-year, TimeWarner posted a profit of $3.36bn, up 27% from its 2003 performance, while revenues grew 6.4% to $42.09bn. \"Our financial performance was strong, meeting or exceeding all of our full-year objectives and greatly enhancing our flexibility,\" chairman and chief executive Richard Parsons said. For 2005, TimeWarner is projecting operating earnings growth of around 5%, and also expects higher revenue and wider profit margins. TimeWarner is to restate its accounts as part of efforts to resolve an inquiry into AOL by US market regulators. It has already offered to pay $300m to settle charges, in a deal that is under review by the SEC. The company said it was unable to estimate the amount it needed to set aside for legal reserves, which it previously set at $500m. It intends to adjust the way it accounts for a deal with German music publisher Bertelsmann's purchase of a stake in AOL Europe, which it had reported as advertising revenue. It will now book the sale of its stake in AOL Europe as a loss on the value of that stake."

import re 
text = re.sub(r'\.','',text)
text = re.sub(r'\'','',text)
text = re.sub(r'\"','',text)
text = re.sub(r'\'','',text)
text = re.sub(r'\%','',text)
text = re.sub(r'\$','',text)
text = re.sub(r'\(','',text)
text = re.sub(r'\)','',text)


# function to test if something one of POS
is_noun = lambda pos: pos[:2] == 'NN'
is_particle = lambda pos: pos[:2] == 'RP'
is_pronoun = lambda pos: ( pos[:2] == 'PR' or pos[:2] == 'WP' )
is_determiner = lambda pos: ( pos[:2] == 'DT' or pos[2:] == 'DT' )
is_verb = lambda pos: pos[:2] == 'VB'
is_adverb = lambda pos: pos[:2] == 'RB'
is_adjective = lambda pos: pos[:2] == 'JJ'
is_conjunction = lambda pos: (pos[:2] == 'CC' or pos[:2]=='IN')
is_interjection = lambda pos: pos[:2] == 'UH'
#do the nlp stuff 



tokenized = nltk.word_tokenize(text)
tagged =  nltk.pos_tag(tokenized)
tagged.append(('END','END'))
tagged.reverse()
tagged.append(('START','START'))
tagged.reverse()

#print tagged
conjunctions = [word for (word,pos) in tagged if is_conjunction(pos)]
interjections = [word for (word,pos) in tagged if is_interjection(pos)]
particles = [word for (word,pos) in tagged if is_particle(pos)]
determiners = [word for (word,pos) in tagged if is_determiner(pos)]
nouns = [word for (word,pos) in tagged if is_noun(pos)]
pronouns = [word for (word,pos) in tagged if is_pronoun(pos)]
verbs = [word for (word,pos) in tagged if is_verb(pos)]
adverbs = [word for (word,pos) in tagged if is_adverb(pos)]
adjectives = [word for (word,pos) in tagged if is_adjective(pos)]

import numpy as np 
nouns = np.unique(nouns).tolist()
conjunctions = np.unique(conjunctions).tolist()
interjections = np.unique(interjections).tolist()
particles = np.unique(particles).tolist()
determiners = np.unique(determiners).tolist()
pronouns = np.unique(pronouns).tolist()
verbs = np.unique(verbs).tolist()
adverbs = np.unique(adverbs).tolist()
adjectives = np.unique(adjectives).tolist()



print '\nThe determiners are :===>', determiners
print '\nThe particles are :===>', particles
print '\nThe conjunctions or prepositions are :===> ', conjunctions
print '\nThe interjections are :===> ', interjections
print '\nThe nouns are :===>' ,  nouns
print '\nThe pronouns are :===>' ,  pronouns
print '\nThe verbs are :===>' , verbs
print '\nThe adverbs are :===>' , adverbs
print '\nThe adjectives are :===>' , adjectives
#############################################################################################
#####using wordnet ##########################################################################
#############################################################################################
#find synonym and antonym of all forms of words 


from nltk.corpus import wordnet as wn 

#("\nEnter which form's synonyms are to be found : ")
#print("\nTop 10 commons nouns are : ")
q = nouns
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas :
        	synonyms.append(l.name)
#	        if l.antonyms() :
#        	    antonyms.append(l.antonyms()[0].name)

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_nouns = forms_frequency.most_common(10)

#############################################################################
#print("\nTop 10 commons adjectives are : ")
q = adjectives
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas :
        	synonyms.append(l.name)
#	        if l.antonyms() :
#        	    antonyms.append(l.antonyms()[0].name)

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_adjectives = forms_frequency.most_common(10)
###################################################################################
#print("\nTop 10 commons pronouns are : ")
q = pronouns
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas :
        	synonyms.append(l.name)
#	        if l.antonyms() :
#        	    antonyms.append(l.antonyms()[0].name)

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_pronouns = forms_frequency.most_common(10)
########################################################################################
#print("\nTop 10 commons verbs are : ")
q = verbs
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas :
        	synonyms.append(l.name)
#	        if l.antonyms() :
#        	    antonyms.append(l.antonyms()[0].name)

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_verbs = forms_frequency.most_common(10)
########################################################################################
#print("\nTop 10 commons adverbs are : ")
q = adverbs
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas :
        	synonyms.append(l.name)
#	        if l.antonyms() :
#        	    antonyms.append(l.antonyms()[0].name)

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_adverbs = forms_frequency.most_common(10)
########################################################################################
#print("\nTop 10 commons Conjunctions are : ")
q = conjunctions
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas :
        	synonyms.append(l.name)
#	        if l.antonyms() :
#        	    antonyms.append(l.antonyms()[0].name)

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_conjunctions = forms_frequency.most_common(10)


##########################################################################################
#print("\nTop 10 commons Interjections are : ")
q = interjections
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas :
        	synonyms.append(l.name)
#	        if l.antonyms() :
#        	    antonyms.append(l.antonyms()[0].name)

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_interjections = forms_frequency.most_common(10)
########################################################################################
#print("\nTop 10 commons particles are : ")
q = particles
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas :
        	synonyms.append(l.name)
#	        if l.antonyms() :
#        	    antonyms.append(l.antonyms()[0].name)

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_particles = forms_frequency.most_common(10)
#######################################################################################
'''Not necessary. It creates complexity in determiners
print("\nTop 10 commons determiners are : ")
q = determiners
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas :
        	synonyms.append(l.name)
#	        if l.antonyms() :
#        	    antonyms.append(l.antonyms()[0].name)

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_determiners = forms_frequency.most_common(10)
'''
print "After using wordnet and getting synonyms : "
#print '\nThe frequent_determiners are :===>', frequent_determiners
print '\nThe frequent_particles are :===>', frequent_particles
print '\nThe frequent_conjunctions or prepositions are :===> ', frequent_conjunctions
print '\nThe frequent_interjections are :===> ', frequent_interjections
print '\nThe frequent_nouns are :===>' ,  frequent_nouns
print '\nThe frequent_pronouns are :===>' ,  frequent_pronouns
print '\nThe frequent_verbs are :===>' , frequent_verbs
print '\nThe frequent_adverbs are :===>' , frequent_adverbs
print '\nThe frequent_adjectives are :===>' , frequent_adjectives






########## LSA on sentence #######################
##LSA is applied on a dataset we can't apply it on a sentence. We need corpus for LSA.
'''
from sklearn.datasets import load_files
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from numpy.linalg import svd
from numpy import *
import pylab
import numpy


bbcdata = text

stemmer = PorterStemmer()
class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
                analyzer = super(StemmedCountVectorizer,self).build_analyzer()
                return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

count_vectorizer = StemmedCountVectorizer(min_df=3, analyzer="word", stop_words=text.ENGLISH_STOP_WORDS)
frequency_matrix = count_vectorizer.fit_transform(bbcdata)
tfidf = bbcdata.TfidfTransformer()



vocabulary = count_vectorizer.fit(bbcdata.data)
#tfidf = TfidfTransformer()


tfidf_matrix = tfidf.fit_transform(frequency_matrix)

svd = TruncatedSVD(n_components = 4)
lsa = make_pipeline(svd, Normalizer(copy=False))
bbc_lsa = lsa.fit_transform(tfidf_matrix)


#To print major features for the compone
for compNum in range(0,4):
        comp = svd.components_[compNum]
        indeces = numpy.argsort(comp).tolist()
        indeces.reverse()
        term = [feat_names[weightIndex] for weightIndex in indeces[0:10]]
        weights = [comp[weightIndex] for weightIndex in indeces[0:10]]
        term.reverse()
        weights.reverse()
        print term
        print weights
        positions = pylab.arange(10)+.5
        pylab.figure(compNum)
        pylab.barh(positions, weights, align='center')
        pylab.yticks(positions, term)
        pylab.xlabel('Weight')
        pylab.title('Strongest terms for component %d' % (compNum))
        pylab.grid(True)
        pylab.savefig('weight_graph'+str(compNum)+'.png')

'''






#######################################################
#HMM part ############################################
#Steps to be done 
# 1. Build HMM on present article 
# 2. Get all patterns for combination of form of word with the synonyms list 
# 3. Get a pattern with good probability from patterns generated in step 2. 


# then all the tag/word pairs for the word/tag pairs in the sentence.
# shorten tags to 2 characters each
sent = tagged
tagged.extend([ (tag[:2], word) for (word, tag) in sent ])



print("++++++++++++This probabiltiy is on original text++++++++++++++++++++++")
#conditional frequency distribution 

cfd_tagsw = nltk.ConditionalFreqDist(tagged)
#print cfd_tags 
# conditional probabiltiy distribution 
cpd_tagsw = nltk.ConditionalProbDist(cfd_tagsw,nltk.MLEProbDist)

print("The probability of an adjective (JJ) being 'gaint' is", cpd_tagsw['JJ'].prob('giant'))
print("The probability of a verb (VB) being 'set' is", cpd_tagsw['VB'].prob('set'))

# Estimating P(ti | t{i-1}) from corpus data using Maximum Likelihood Estimation (MLE):
# P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
text_tags = [tag for (tag, word) in sent ]

#print text_tags

# make conditional frequency distribution:
# count(t{i-1} ti)
cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(text_tags))
# make conditional probability distribution, using
# maximum likelihood estimate:
# P(ti | t{i-1})
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)
print("If we have just seen 'DT', the probability of 'NN' is", cpd_tags["DT"].prob("NN"))
print( "If we have just seen 'VB', the probability of 'JJ' is", cpd_tags["VB"].prob("DT"))
print( "If we have just seen 'VB', the probability of 'NN' is", cpd_tags["VB"].prob("NN"))

