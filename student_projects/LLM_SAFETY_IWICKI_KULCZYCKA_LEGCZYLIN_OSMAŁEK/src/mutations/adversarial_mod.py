import random

substitution_map = {
	"a": ["o"],
	"b": ["d"],
	"c": ["s"],
	"d": ["b"],
	"e": [],
	"f": [],
	"g": ["q", "p"],
	"h": ["n"],
	"i": ["l", "j"],
	"j": ["i"],
	"k": [],
	"l": ["i"],
	"m": ["n"],
	"n": ["m", "h"],
	"o": ["a"],
	"p": ["q", "g"],
	"q": ["p", "g"],
	"r": [],
	"s": ["z"],
	"t": [],
	"u": ["v"],
	"v": ["u"],
	"w": [],
	"x": [],
	"y": [],
	"z": ["s"],
}


def misspell(word: str, remove_prob=0.05, duplicate_prob=0.2, substitute_prob=0.2):
	""" Misspells a word with three modifications, every occurring at most once: removes, duplicates or substitutes a character """
	
	result = []
	
	word = word.lower()
	
	removed = False
	duplicated = False
	substituted = False
	
	for c in word:
		if random.random() < remove_prob and not removed:
			removed = True
			continue
		elif random.random() < duplicate_prob and not duplicated:
			result.append(c)
			result.append(c)
			duplicated = True
			continue
		elif random.random() < substitute_prob and not substituted:
			if len(substitution_map[c]) != 0:
				result.append(random.choice(substitution_map[c]))
				substituted = True
				continue
				
		result.append(c)
	
	return "".join(result)


unicode_look_alike_map = {
	"a": ["а", "ạ", "ą", "ä", "à", "á", "ą"],
	"b": [],
	"c": ["с", "ƈ", "ċ"],
	"d": ["ԁ", "ɗ"],
	"e": ["е", "ẹ", "ė", "é", "è"],
	"f": [],
	"g": ["ġ"],
	"h": ["һ"],
	"i": ["і", "í", "ï"],
	"j": ["ј", "ʝ"],
	"k": ["κ"],
	"l": ["ӏ", "ḷ"],
	"m": [],
	"n": ["ո"],
	"o": ["о", "ο", "օ", "ȯ", "ọ", "ỏ", "ơ", "ó", "ò", "ö"],
	"p": ["р"],
	"q": ["զ"],
	"r": [],
	"s": ["ʂ"],
	"t": [],
	"u": ["υ", "ս", "ü", "ú", "ù"],
	"v": ["ν", "ѵ"],
	"w": [],
	"x": ["х", "ҳ"],
	"y": ["у", "ý"],
	"z": ["ʐ", "ż"],
}


def mix_unicode_char(word: str, substitute_prob=0.2):
	""" Changes some letters to their Unicode look-alike equivalents """
	
	result = []
	
	word = word.lower()
	
	for c in word:
		if random.random() < substitute_prob:
			if len(unicode_look_alike_map[c]) != 0:
				result.append(random.choice(unicode_look_alike_map[c]))
				continue
				
		result.append(c)
	
	return "".join(result)


def syntax_permutation(sentence: str):
	""" Don't really know how to tackle this one """
	return


if __name__ == '__main__':
	print(mix_unicode_char("credit"))
