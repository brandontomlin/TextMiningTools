from BeautifulSoup import BeautifulSoup
import urllib2
import os
import imp

import pandas as pd

from lxml import html
from natsort import natsorted



cities = 'https://www.craigslist.org/about/sites'

page = urllib2.urlopen(cities)
soup = BeautifulSoup(page.read())
soup.prettify()


uls1 = soup.findAll("div", attrs= {"class":"box box_1"})
uls2 = soup.findAll("div", attrs= {"class":"box box_2"})
uls3 = soup.findAll("div", attrs= {"class":"box box_3"})
uls4 = soup.findAll("div", attrs= {"class":"box box_4"})



b1 = lis = [li for ul in uls1 for li in ul.findAll('a', href=True)]
b2 = lis = [li for ul in uls2 for li in ul.findAll('a', href=True)]
b3 = lis = [li for ul in uls2 for li in ul.findAll('a', href=True)]
b4 = lis = [li for ul in uls2 for li in ul.findAll('a', href=True)]


##########

# SORT B1-4 BY .ORG #

##########
one = []
for i in b1:
	one.append(i['href'])

print natsorted(one)
# for i in b2:
# 	print i['href']

# for i in b3:
# 	print i['href']

# for i in b4:
# 	print i['href']



##########
##########
##########
##########


def NextPage():
	url = "http://kansascity.craigslist.org/search/hva"

	page = urllib2.urlopen(url)
	soup = BeautifulSoup(page.read())
	soup.prettify()

	titleSpan = soup.findAll("span", attrs = {"id":"titletextonly"})

	headlines = [] 
	for row in titleSpan:
	    headlines.append(row.getText())
	    
	columns = ['headline']

	df = pd.DataFrame(headlines, 
	                 columns = columns)


	# Find next page and loop to end. 
	## Needs completion. ( Combine URL to next page with page number. )

	nextPage = soup.findAll('a', attrs={'class':'button next'})

	test =[]
	for row in nextPage:
	    test.append(row['href'])
	    
	del test[:1]
	test = str(test)
	test = test.replace('100','')
	type(test)

	listFull = []

	increment = 100
	counter = 0 
	while counter < 1800:
	        counter += increment
	        listFull.append(counter)

	print counter

	for i in listFull:
	    test = ''.join(map(str, test))

	    print (test, i)