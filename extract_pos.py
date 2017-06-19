#from sklearn.datasets import load_files

import nltk
text = "Hey goody Alice and Ram, we are going outside. Would you like to come with us or sit with smart John ? Anyways the decision is yours. Your level of understanding is higher than us!!! We will come back very quick. Goodbye ! See you later."
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
print tagged
conjunctions = [word for (word,pos) in tagged if is_conjunction(pos)]
interjections = [word for (word,pos) in tagged if is_interjection(pos)]
particles = [word for (word,pos) in tagged if is_particle(pos)]
determiners = [word for (word,pos) in tagged if is_determiner(pos)]
nouns = [word for (word,pos) in tagged if is_noun(pos)]
pronouns = [word for (word,pos) in tagged if is_pronoun(pos)]
verbs = [word for (word,pos) in tagged if is_verb(pos)]
adverbs = [word for (word,pos) in tagged if is_adverb(pos)]
adjectives = [word for (word,pos) in tagged if is_adjective(pos)]
print '\nThe determiners are :===>', determiners
print '\nThe particles are :===>', particles
print '\nThe conjunctions or prepositions are :===> ', conjunctions
print '\nThe interjections are :===> ', interjections
print '\nThe nouns are :===>' ,  nouns
print '\nThe pronouns are :===>' ,  pronouns
print '\nThe verbs are :===>' , verbs
print '\nThe adverbs are :===>' , adverbs
print '\nThe adjectives are :===>' , adjectives


#######################################################
#HMM part ############################################

#from nltk.corpus import brown 
#conditional frequency distribution 
cfd_tagsw = nltk.ConditionalFreqDist(tagged)
#print cfd_tags 
# conditional probabiltiy distribution 
cpd_tagsw = nltk.ConditionalProbDist(cfd_tagsw,nltk.MLEProbDist)









