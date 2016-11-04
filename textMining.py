import urllib2
from BeautifulSoup import BeautifulSoup
from nltk import nltk 


pd.set_option('display.max_colwidth', 300)

def requesturltoSoup(url, parseUrls):  
	page = urllib2.urlopen(url).read()
	soup = BeautifulSoup(page)
	soup.prettify()
	for anchor in soup.findAll(parseUrls, href=True):
	    print anchor['href']
	return soup 


def organizesoup(soup, category, tag, _id, classifier):
    BSobj = soup.find_all(tag, {_id: classifier})
    headlines = {}
    for name in BSobj:
        headlines[name.get_text()] = {(time.strftime("%m/%d/%Y")), 
                                      category}
    return headlines

def findtags(tag_prefix, tag_text):
	cfd = nltk.ConditonalFreqDist((tag, word) for (word, tag) in tag_text
		if tag.startswith(tag_prefix))
	return dict((tag, cfd[tag].keys()[:5]) for tag in cfd.conditions())

def process(sentence):
	# Consider each three-words window in the sentence
	# and if they meet our criteria.
	for (w1, t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
		if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')):
			print w1, w2, w3

# for tagged_sent in brown.tagged_sents():
# 	process(tagged_sents)
	