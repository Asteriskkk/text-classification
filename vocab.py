import json 
import pandas as pd
# here file.json is your json file 


class Vocab:
	def __init__(self, vocab):
		self.vocab = vocab

	def __len__(self):
		return len(self.vocab[1])

	def __getitem__(self,key):
		return self.vocab[key]
