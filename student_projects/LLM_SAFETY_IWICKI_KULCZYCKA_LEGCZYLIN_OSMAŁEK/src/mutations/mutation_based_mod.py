import random
from itertools import chain

import nltk

nltk.download('wordnet')

from nltk.corpus import wordnet


def get_synonyms(word: str):
	synonyms = wordnet.synsets(word)
	
	return list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))


def get_random_synonym(word: str):
	return random.choice(get_synonyms(word))


if __name__ == '__main__':
	print(get_random_synonym("get"))
