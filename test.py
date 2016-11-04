import urllib2
from BeautifulSoup import BeautifulSoup
from pandas import pandas as pd

url = 'http://www.archives.gov/exhibits/featured_documents/magna_carta/translation.html'

def requesturltoSoup(url, parseUrls):  
        page = urllib2.urlopen(url).read()
        soup = BeautifulSoup(page)
        soup.prettify()
        text = []
        for anchor in soup.findAll(parseUrls ):
            text.append(anchor)
        return soup, text

soup, text = requesturltoSoup(url, 'p')

print text

