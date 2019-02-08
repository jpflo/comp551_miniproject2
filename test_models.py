import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.decomposition import PCA

from src.models.custom_models import CustomBernoulliNaiveBayes
from src.data.make_dataset import raw_data_extraction

from sklearn.feature_extraction.text import CountVectorizer

import time


directories = ['data/raw/train/neg/', 'data/raw/train/pos/']
raw_text_lst, raw_target_lst, raw_text_id = raw_data_extraction(directories)
raw_data = pd.DataFrame({'target':raw_target_lst, 'raw_text':raw_text_lst})
raw_data.to_pickle('data/interim/extracted_training_text')

## reduce how many examples we use, if wanted (should run fine with full data set)
data = raw_data.sample(frac=1.0)

## count words and binarize (1 if word appears, 0 if not)
corpus = data['raw_text'].to_numpy()
vectorizer = CountVectorizer()
word_counts_raw = vectorizer.fit_transform(corpus)
word_counts = word_counts_raw#.todense()

## some extra features for future
#vectorizer = TfidfVectorizer()
#data = count_past_tense_verbs(data)

## set up train/valid data
y = data['target'].to_numpy()
X = word_counts.astype(bool)

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

## initialize the model
clf = CustomBernoulliNaiveBayes(laplaceSmoothing = True)  # our written model, can use all the SKLearn functions with it (see src/models/custom_models for the code)
#clf = DecisionTreeClassifier(random_state=0)
#clf = SVC(gamma=0.001, C=100.)
#clf = BernoulliNB()  # to compare our custom model with - we get identical results, but slightly slower

## train once and predict
t1 = time.time()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print('Score: {}'.format(score))

## do k-fold cross-validation
#cv = cross_val_score(clf, X, y, cv = 5)
#print('Mean for cross-validation: {}, Individual: {}'.format(np.mean(cv), cv))

t2 = time.time()
print('Time to train and predict/cross-validate: {}'.format(t2-t1))