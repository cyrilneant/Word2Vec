# "Word2vec, NLP with Deep Learning"

import numpy as np 
import json
import re
import datetime as datetime
import theano
import theano.tensor as T 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle

import os 
import system
import pandas as pd 

# load your data here

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# init_weights ==> initial weights with small value

def init_weights(shape):
	return np.random.randn(*shape).astype(np.float32) / np.sqrt(sum(shape))

class Model:
	def __init__(self, D, V, context_szq):		
		self.D = D
		self.V = V
		self.context_sz = context_sz

	def _get_pnw(self, X):
		word_freq = {}
		word_count = sum(len(x) for x in X)
		for x in X:
			for xj in x:
				if xj not in x:
					word_freq[xj] = 0
				word_freq[xj] += 1
		self.Pnw = np.zeros(self.V)
		for j in range(2, self.V):
			self.Pnw[j] = (word_freq[j] / float(word_count))**0.75
		assert(np.all(self.Pnw[2:] > 0))
		return self.Pnw

	def _get_negative_samples(self, context, num_neg_samples):
		saved = {}
		for context_idx in context:
			saved[context_idx] = self.Pnw[context_idx]
			self.Pnw[context_idx] = 0
		neg_samples = np.random.choice(
			range(self.V),
			size=num_neg_samples,
			replace=false,
			p=self.Pnw / np.sum(self.Pnw)
			)
		for j, pnwj in saved.iteritems():
			self.Pnw[j] = pnwj
		assert(np.all(self.Pnw[j] > 0))
		return neg_samples 

	def fit()