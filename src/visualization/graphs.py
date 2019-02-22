# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:39:39 2019

@author: jpflo
"""

#%% Import Data
import csv
import numpy as np
import matplotlib.pyplot as plt

with open('model_tests_results.csv', newline='') as csvfile:
    raw_data = list(csv.reader(csvfile))
    
# data = np.array(data[1:])
data = raw_data[1:]

msv = [float(row[-2]) for row in data]
error = [float(row[-1]) for row in data]
labels = []

for d in data:
    str = d[0] + ' \n ' + d[1] + ' \n '
    
    if d[2] == '(1,1)':
        str += 'Unigrams \n '
    else:
        str += 'Unigrams and Bigrams \n '
    
    if data[3]:
        str += 'TFIDF'
    else:
        str += 'No TFIDF'
    
    labels.append(str)

print(data)


#%%

x = range(len(msv))

plt.bar(labels, list(msv),yerr=error)


