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

text = text.decode('utf-8')
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
#	for syn in wn.synsets(word_q) :
# the loop is to be noticed. It uses new nltk
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas() :
        	synonyms.append(str(l.name()))

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_nouns = forms_frequency.most_common(10)
similar_nouns = forms_frequency

#############################################################################
#print("\nTop 10 commons adjectives are : ")
q = adjectives
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas() :
        	synonyms.append(str(l.name()))

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_adjectives = forms_frequency.most_common(10)
similar_adjectives = forms_frequency
###################################################################################
#print("\nTop 10 commons pronouns are : ")
q = pronouns
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas() :
        	synonyms.append(str(l.name()))

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_pronouns = forms_frequency.most_common(10)
similar_pronouns = forms_frequency

########################################################################################
#print("\nTop 10 commons verbs are : ")
q = verbs
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas() :
        	synonyms.append(str(l.name()))

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_verbs = forms_frequency.most_common(10)
similar_verbs = forms_frequency
########################################################################################
#print("\nTop 10 commons adverbs are : ")
q = adverbs
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas() :
        	synonyms.append(str(l.name()))

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_adverbs = forms_frequency.most_common(10)
similar_adverbs = forms_frequency
########################################################################################
#print("\nTop 10 commons Conjunctions are : ")
q = conjunctions
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas() :
        	synonyms.append(str(l.name()))

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_conjunctions = forms_frequency.most_common(10)
similar_conjunctions = forms_frequency

##########################################################################################
#print("\nTop 10 commons Interjections are : ")
q = interjections
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas() :
        	synonyms.append(str(l.name()))

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_interjections = forms_frequency.most_common(10)
similar_interjections = forms_frequency
########################################################################################
#print("\nTop 10 commons particles are : ")
q = particles
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas() :
        	synonyms.append(str(l.name()))

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_particles = forms_frequency.most_common(10)
similar_particles = forms_frequency
#######################################################################################
#Not necessary. It creates complexity in determiners
#print("\nTop 10 commons determiners are : ")
q = determiners
synonyms = []
antonyms = []
for word_q in q :
	for syn in wn.synsets(word_q) :
	    for l in syn.lemmas() :
        	synonyms.append(str(l.name()))

#synonyms = np.unique(synonyms).tolist()
#print("Synonyms are of ",q , "are ==> \n",synonyms)
from collections import Counter
forms_frequency = Counter(synonyms)
#print forms_frequency
#print "Most common are : "
#print forms_frequency.most_common(10)
frequent_determiners = forms_frequency.most_common(10)
similar_determiners = forms_frequency


#############################Printing stuff###############################
###frequent synonyms (top 10)#####
#print '\nThe frequent_determiners are :===>', frequent_determiners
#print '\nThe frequent_particles are :===>', frequent_particles
#print '\nThe frequent_conjunctions or prepositions are :===> ', frequent_conjunctions
#print '\nThe frequent_interjections are :===> ', frequent_interjections
#print '\nThe frequent_nouns are :===>' ,  frequent_nouns
#print '\nThe frequent_pronouns are :===>' ,  frequent_pronouns
#print '\nThe frequent_verbs are :===>' , frequent_verbs
#print '\nThe frequent_adverbs are :===>' , frequent_adverbs
#print '\nThe frequent_adjectives are :===>' , frequent_adjectives


#print "After using wordnet and getting synonyms : "
#print '\nThe simlar_determiners are :===>', similar_determiners
#print '\nThe similar_particles are :===>', similar_particles
#print '\nThe similar_conjunctions or prepositions are :===> ', similar_conjunctions
#print '\nThe similar_interjections are :===> ', similar_interjections
#print '\nThe similar_nouns are :===>' ,  similar_nouns
#print '\nThe similar_pronouns are :===>' ,  similar_pronouns
#print '\nThe similar_verbs are :===>' , similar_verbs
#print '\nThe similar_adverbs are :===>' , similar_adverbs
#print '\nThe similar_adjectives are :===>' , similar_adjectives

#######################Tagging on similar forms #########################################


#############tag nouns #############################################
index = 0
wl = similar_nouns
tagged_similar_nouns = []
for w in wl :
    tagged_similar_nouns.append((w,'NN'))
    index = index +1
#print "tagged_similar_nouns", tagged_similar_nouns
#######################tag determiners############
index = 0
wl = similar_determiners
tagged_similar_determiners = []
for w in wl :
    tagged_similar_determiners.append((w,'DT'))
    index = index +1
#print '\n tagged_similar_determiners',tagged_similar_determiners
#######################tag particles 
index = 0
wl = similar_particles
tagged_similar_particles = []
for w in wl :
    tagged_similar_particles.append((w,'RP'))
    index = index +1
#print '\n tagged_similar_particles',tagged_similar_particles
#######################tag conjunctions or prepositions 
index = 0
wl = similar_conjunctions
tagged_similar_conjunctions = []
for w in wl :
    tagged_similar_conjunctions.append((w,'CC'))
    index = index +1
#print '\ntagged_similar_conjunctions',tagged_similar_conjunctions
#######################tag interjections 
index = 0
wl = similar_interjections
tagged_similar_interjections = []
for w in wl :
    tagged_similar_interjections.append((w,'UH'))
    index = index +1
#print '\ntagged_similar_interjections',tagged_similar_interjections
#######################tag pronouns 
index = 0
wl = similar_pronouns
tagged_similar_pronouns = []
for w in wl :
    tagged_similar_pronouns.append((w,'PR'))
    index = index +1
#print '\ntagged_similar_pronouns',tagged_similar_pronouns
#######################tag verbs 
index = 0
wl = similar_verbs
tagged_similar_verbs = []
for w in wl :
    tagged_similar_verbs.append((w,'VB'))
    index = index +1
#print '\n',tagged_similar_verbs
#######################tag adverbs 
index = 0
wl = similar_adverbs
tagged_similar_adverbs = []
for w in wl :
    tagged_similar_adverbs.append((w,'RB'))
    index = index +1
#print '\ntagged_similar_adverbs',tagged_similar_adverbs
#######################tag adjectives 
index = 0
wl = similar_adjectives
tagged_similar_adjectives = []
for w in wl :
    tagged_similar_adjectives.append((w,'JJ'))
    index = index +1
#print '\ntagged_similar_adjectives',tagged_similar_adjectives



#######################################################
#HMM part ############################################
#Steps to be done 
# 1. Build HMM on present article 
# 2. Get all patterns for combination of form of word with the synonyms list 
# 3. Get a pattern with good probability from patterns generated in step 2. 


# then all the tag/word pairs for the word/tag pairs in the sentence.
# shorten tags to 2 characters each
sent = tagged
tagged.append(('END','END'))
tagged.reverse()
tagged.append(('START','START'))
tagged.reverse()

tagged.extend([ (tag[:2], word) for (word, tag) in sent ])

print tagged

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


print("++++++++++++++++++++++HMM on forms +++++++++++++++++++++++++++++++++++")

import itertools 

tagged = list(itertools.chain(tagged_similar_adjectives,tagged_similar_verbs,tagged_similar_adverbs, tagged_similar_pronouns, tagged_similar_interjections, tagged_similar_conjunctions, tagged_similar_particles, tagged_similar_determiners, tagged_similar_nouns))
tagged.append(('END','END'))
tagged.reverse()
tagged.append(('START','START'))
tagged.reverse()
sent = tagged
tagged.extend([ (tag[:2], word) for (word, tag) in sent ])
print "\n============\n"
print tagged
print "\n============\n"
#conditional frequency distribution 
cfd_tagsw = nltk.ConditionalFreqDist(tagged)
#print cfd_tags 
# conditional probabiltiy distribution 
cpd_tagsw = nltk.ConditionalProbDist(cfd_tagsw,nltk.MLEProbDist)

print("The probability of an Noun (NN) being 'stockpile' is", cpd_tagsw['NN'].prob('stockpile'))
print("The probability of a Adjective (JJ) being 'justify' is", cpd_tagsw['JJ'].prob('justify'))

# Estimating P(ti | t{i-1}) from corpus data using Maximum Likelihood Estimation (MLE):
# P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
text_tags = [tag for (tag, word) in sent ]

#print text_tags

# make conditional frequency distribution:
# count(t{i-1} ti)
cfd_tagsw= nltk.ConditionalFreqDist(nltk.bigrams(text_tags))
# make conditional probability distribution, using
# maximum likelihood estimate:
# P(ti | t{i-1})
cpd_tagsw = nltk.ConditionalProbDist(cfd_tagsw, nltk.MLEProbDist)

print("If we have just seen 'DT', the probability of 'NN' is", cpd_tagsw["DT"].prob("NN"))
print( "If we have just seen 'VB', the probability of 'JJ' is", cpd_tagsw["VB"].prob("DT"))
print( "If we have just seen 'VB', the probability of 'NN' is", cpd_tagsw["VB"].prob("NN"))
###
# putting things together:
# what is the probability of the tag sequence "PP VB TO VB" for the word sequence "I want to race"?
# It is
# P(START) * P(PP|START) * P(I | PP) *
#            P(VB | PP) * P(want | VB) *
#            P(TO | VB) * P(to | TO) *
#            P(VB | TO) * P(race | VB) *
#            P(END | VB)
#
# We leave aside P(START) for now.

#here we will have to replace string "I want to race" by new strings that we can get from combination of all forms of a word

prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tagsw["PP"].prob("I") * \
    cpd_tags["PP"].prob("VB") * cpd_tagsw["VB"].prob("want") * \
    cpd_tags["VB"].prob("TO") * cpd_tagsw["TO"].prob("to") * \
    cpd_tags["TO"].prob("VB") * cpd_tagsw["VB"].prob("race") * \
    cpd_tags["VB"].prob("END")


print( "The probability of the tag sequence 'START PP VB PP NN END' for 'I saw her duck' is:", prob_tagsequence)

prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tagsw["PP"].prob("I") * \
    cpd_tags["PP"].prob("VB") * cpd_tagsw["VB"].prob("saw") * \
    cpd_tags["VB"].prob("PP") * cpd_tagsw["PP"].prob("her") * \
    cpd_tags["PP"].prob("VB") * cpd_tagsw["VB"].prob("duck") * \
    cpd_tags["VB"].prob("END")

print( "The probability of the tag sequence 'START PP VB PP VB END' for 'I saw her duck' is:", prob_tagsequence)


# Now implement virtebi algorithm
distinct_tags = {'NN','RP','PR','WP','DT','VB','RB','JJ','CC','IN','UH'}

#sentence = ["I", "want", "to", "race" ]
#sentence = ["I", "saw", "her", "duck" ]
# here for some entries we are getting tags as words 
sentence = [word for (tag, word) in sent ]
print sentence
sentlen = len(sentence)

# viterbi:
# for each step i in 1 .. sentlen,
# store a dictionary
# that maps each tag X
# to the probability of the best tag sequence of length i that ends in X
viterbi = [ ]

# backpointer:
# for each step i in 1..sentlen,
# store a dictionary
# that maps each tag X
# to the previous tag in the best tag sequence of length i that ends in X
backpointer = [ ]

first_viterbi = { }
first_backpointer = { }
for tag in distinct_tags:
    # don't record anything for the START tag
    if tag == "START": continue
    first_viterbi[ tag ] = cpd_tags["START"].prob(tag) * cpd_tagsw[tag].prob( sentence[0] )
    first_backpointer[ tag ] = "START"

print(first_viterbi)
print(first_backpointer)
    
viterbi.append(first_viterbi)
backpointer.append(first_backpointer)

currbest = max(first_viterbi.keys(), key = lambda tag: first_viterbi[ tag ])
print( "Word", "'" + sentence[0] + "'", "current best two-tag sequence:", first_backpointer[ currbest], currbest)
# print( "Word", "'" + sentence[0] + "'", "current best tag:", currbest)

for wordindex in range(1, len(sentence)):
    this_viterbi = { }
    this_backpointer = { }
    prev_viterbi = viterbi[-1]
    
    for tag in distinct_tags:
        # don't record anything for the START tag
        if tag == "START": continue

        # if this tag is X and the current word is w, then 
        # find the previous tag Y such that
        # the best tag sequence that ends in X
        # actually ends in Y X
        # that is, the Y that maximizes
        # prev_viterbi[ Y ] * P(X | Y) * P( w | X)
        # The following command has the same notation
        # that you saw in the sorted() command.
        best_previous = max(prev_viterbi.keys(),
                            key = lambda prevtag: \
            prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagsw[tag].prob(sentence[wordindex]))

        # Instead, we can also use the following longer code:
        # best_previous = None
        # best_prob = 0.0
        # for prevtag in distinct_tags:
        #    prob = prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex])
        #    if prob > best_prob:
        #        best_previous= prevtag
        #        best_prob = prob
        #
        this_viterbi[ tag ] = prev_viterbi[ best_previous] * \
            cpd_tags[ best_previous ].prob(tag) * cpd_tagsw[ tag].prob(sentence[wordindex])
        this_backpointer[ tag ] = best_previous

    currbest = max(this_viterbi.keys(), key = lambda tag: this_viterbi[ tag ])
    print( "Word", "'" + sentence[ wordindex] + "'", "current best two-tag sequence:", this_backpointer[ currbest], currbest)
    # print( "Word", "'" + sentence[ wordindex] + "'", "current best tag:", currbest)


    # done with all tags in this iteration
    # so store the current viterbi step
    viterbi.append(this_viterbi)
    backpointer.append(this_backpointer)


# done with all words in the sentence.
# now find the probability of each tag
# to have "END" as the next tag,
# and use that to find the overall best sequence
prev_viterbi = viterbi[-1]
best_previous = max(prev_viterbi.keys(),
                    key = lambda prevtag: prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob("END"))

prob_tagsequence = prev_viterbi[ best_previous ] * cpd_tags[ best_previous].prob("END")

# best tagsequence: we store this in reverse for now, will invert later
best_tagsequence = [ "END", best_previous ]
# invert the list of backpointers
backpointer.reverse()

# go backwards through the list of backpointers
# (or in this case forward, because we have inverter the backpointer list)
# in each case:
# the following best tag is the one listed under
# the backpointer for the current best tag
current_best_tag = best_previous
for bp in backpointer:
    best_tagsequence.append(bp[current_best_tag])
    current_best_tag = bp[current_best_tag]

best_tagsequence.reverse()
#print( "The sentence was:", end = " ")
print( "The sentence was:") #, end = " ")
print( "The sentence was:")
#for w in sentence: print( w, end = " ")
for w in sentence: print( w)
print("\n")
#print( "The best tag sequence is:", end = " ")
print( "The best tag sequence is:")
#for t in best_tagsequence: print (t, end = " ")
for t in best_tagsequence: print (t)
print("\n")
print( "The probability of the best tag sequence is:", prob_tagsequence)

