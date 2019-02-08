import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import nltk as nl
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

from nltk.corpus import wordnet


stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()


def convert_pos_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None



def count_past_tense_verbs(data):
    n_past_tense_verbs = []
    
    for text in data['raw_text']:
        tokens = word_tokenize(text) # Generate list of tokens
        tokens_pos = pos_tag(tokens) 
        verb_past_counter = 0
        for token_pair in tokens_pos:
#            word = token_pair[0]
            tag = token_pair[1]
#            pos = convert_pos_tag(tag)
            
            if tag == 'VBD':
                verb_past_counter += 1
            
#            if pos is not None:
#                lemma = lemmatiser.lemmatize(word, pos=pos)
        n_past_tense_verbs.append(verb_past_counter)
        
    data['n_past_tense_verbs'] = n_past_tense_verbs
    
    return data