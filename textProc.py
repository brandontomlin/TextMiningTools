from nltk import nltk

def lexical_diversity(text):

	#
	
	return len(text)  len(set(text))

def lexical_diversityScore(text):

	#

	wordCount = len(text)
	vocabSize = len(text)

	diversityScore = wordCount / vocabSize
	return diversityScore

def plural(word):

	# returns the plural form of a word

	if word.endswith('y'):

		return word[:-1] + 'ies'

	elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
		return word + "es"

	elif word.endswith('an'):

		return word[:2] + 'en'
	else: 
		return word + 's'

def unusual_words(text):

	#

	text_vocab       = set(w.lower() for w in text if w.isalpha())
	english_vocab  = set(w.lower() for w in nltk.corpus.words.words())
	unusual            = text_vocab.difference(english_vocab)
	return sorted(unusual)

def content_fraction(text):

	# 

	stopwords = nltk.corpus.stopwords.words('english')
	content = [w for w in text if w.lower() not in stopwords]
	return len(content) / len(text)


from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk import FreqDist
import string

sw = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def isStopWord(word):
    return word in sw or word in punctuation

data = data

filtered = [w.lower() for w in data not isStopWord(w.lower())]


def multiple_replace(dict, text): 

  """ Replace in 'text' all occurences of any key in the given
  dictionary by its corresponding value.  Returns the new tring.""" 
  text = str(text).lower()

  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


measurementDic = {'girl'  : 'female',
                  'girls' : 'female',
                  'woman' : 'female',
                  'women' : 'female',
                  
                  'gentlemen': 'males',
                  'man'    : 'male',
                  'men'    : 'male',
                  'boy'    : 'male',
                  'boys'   : 'male',
                  
                  'inches'  : 'distanceMeasurement',
                  'miles'   : 'distanceMeasurement',
                  'yards'   : 'distanceMeasurement',
                  'feet'    : 'distanceMeasurement',
                  'foot'    : 'distanceMeasurement',
                  'light year' 
                             : 'distanceMeasurement ',
                  'hours'    : 'timeMeasurement ',
                  'hour'     : 'timeMeasurement ',
                  'minute'   : 'timeMeasurement ',
                  'minutes'  : 'timeMeasurement ',
                  'second'   : 'timeMeasurement ',
                  'seconds'  : 'timeMeasurement '}


df['removedFullText'] = map(lambda x: multiple_replace(measurementDic, x), df['cleanedFullText'])



# Regular expression Tagger

patterns = [ (r'.*ing$', ' VBG' ),
		(r'.*ed*', 'VBD'),
		(r'.*es$', 'VBZ'),
		(r'.*ould$', 'MD'),
		(r'.*\'s$', 'NN$',),
		(r'.*s$', 'NNS'),
		(r'^-?[0-9]+(.[0-9]+?$', 'CD')),
		(r'.*', 'NN'),
]

regexp_tagger = nltk.regexpTaggerpatterns)
regexp_tagger.tag(brown_sents[3])
regexp_tagger.evaluate(brown_tagged_sents )

# Look up tagger performance with varying model size

def performance(cfd, wordlist):
	lt = dict((word, cfd[word].max()) for word in wordlist)
	baseline_tagger = nltk.UnigramTagger(model=lt, backoff = nltk.DefaultTagger('NN'))
	return baseline_tagger.evaluate(brown.brown_tagged_sents(categories = 'news'))
def display():
	import pylab
	words_by_freq = list(nltk.FreqDist(brown.words(categories='news')))
	cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
	size = 2 * pylab.arange(15)
	pefs = [performance(cfd, words_by_freq[:size]) for size in sizes]
	pylab.plot(sizes, perfs, '-bo')
	pylab.title('')
	pylab.xlabel('')
	pylab.ylabel('')
	pylab.show()

display()

# Unigram tagger 

from nltk.corpus import brown
brown_tagged_sents = brown_tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2007])

# Unigram tagger separating the training and testing data 

size = int(len(brown_sents)*0.9)
size 
training_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(training_sents)
unigram_tagger.evaluate(test_sents)

bigram_tagger = nltk.BigramTagger(training_sents)
bigram_tagger.tag(brown_sents[2007])



# Combining Taggers 


t0 =  nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(training_sents, backoff = t0)
t2 = BigramTagger(training_sents, cutoff = 2, backoff = t1)
t2.evaluate(test_sents)
t3 = TrigamTagger(training_sents, cutoff = 2, backoff = t2)
t3.evaluate(test_sents)

## Storing Taggers

from cPickle import dump
output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()

from cPickle import load
input_ = open('t2.pkl', 'rb')
tagger  = load(input_)
input_.close()

cfd = nltk.ConditionalFreqDist((x[1], y[1], z[0], z[1]
	for sent in brown_tagged_sents
	for x, y, z, in nltk.trigrams(sent))
ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]>1)
sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N()

## Confusion Matrix 

test_tags = [tag for sent in brown_sents(categories='editorial')
	for (word, tag) in t2.tag(sent)]
gold_tags = [tag for (word, tag) in brown_tagged_words(categories='editorial')]
print nltk.ConfusionMatrix(gold, test)


# Tagging Across Sentence Boundaries

brown_tagged_sents = brown_tagged_words(categories = 'news')
brown_sents = brown_sents(categories = 'news')

size = int(len(brown_tagged_sents) * 0.9)
training_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

t0 =  nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(training_sents, backoff = t0)
t2 = BigramTagger(training_sents, cutoff = 2, backoff = t1)
t2.evaluate(test_sents)

























