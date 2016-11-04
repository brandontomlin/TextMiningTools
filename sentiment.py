def hl_sent(inputstring):

    poscount = 0
    negcount = 0
    i = 0


    for word in inputstring.split():
        if i > 0:
            prev = inputstring.split().pop(i-1)
        else:
            prev =""

        if HLpos.count(word):
            if negate.count(prev):
                negcount += 1
            elif amplify.count(prev):
                poscount +=2
            else: 
                poscount +=1
        elif HLneg.count(word):
            if negate.count(prev):
                poscount += 1
            elif amplify.count(prev):
                negcount +=2
            else:
                negcount +=1
        i+=1
    
    if poscount+negcount > 0:
        t = float((poscount - negcount)/(poscount+negcount))
        
    else:
        t = 0
    
    
    if t > 0:
        tone = "Positive"
    elif t < 0:
        tone = "Negative"
    else:
        tone = "Neutral"
    
    return tone


#amplification and negation words from qdap
negate = ["aint", "arent","cant", "couldnt" , "didnt" , "doesnt" ,"dont" ,
			"hasnt" , "isnt" ,"mightnt" , "mustnt" ,"neither" ,"never", "no",
			"nobody" , "nor", "not" , "shant", "shouldnt", "wasnt" , "werent",
			"wont", "wouldnt"]
amplify = ["acute" ,"acutely", "certain", "certainly" ,"colossal", "colossally",
			"deep" , "deeply" , "definite","definitely" ,"enormous","enormously" ,
			 "extreme", "extremely" ,"great","greatly" ,"heavily", "heavy", "high",
			 "highly" ,"huge","hugely" , "immense", "immensely" ,"incalculable",
			 "incalculably","massive", "massively", "more","particular" ,"particularly",
			 "purpose", "purposely", "quite" ,"real" ,"really","serious", "seriously", 
			 "severe","severely" ,"significant" ,"significantly","sure","surely" , "true" ,
			"truly" ,"vast" , "vastly" , "very"]


#if you want to verify that a word is in the dictionary
#create a dictionary list dictionary list
import string
exclude = set(string.punctuation)
#download word list from http://app.aspell.net/create to a text file and parse in next line
wordlist = [line.lower().strip() for line in open('path name for text file from SCOWL', 'r')]
wordlist = map(lambda x: (''.join(ch for ch in x if ch not in exclude)), wordlist)

# to use, just check the number of time the word appears in wordlist:
# if you want to see if "checkword" is in your list
print (wordlist.count(checkword))


# code for parsing Loughran and McDonald dictionaries. 

def finance_sent(inputstring):
    
    poscount = 0
    negcount = 0
    
    

    for word in inputstring.split():
        #print word
        if word in dictlist2:
            blah = dictlist2.index(word)
            negcount = negcount + int(smalldict.iloc[blah]['Negative'] !=0)
            poscount = poscount + int(smalldict.iloc[blah]['Positive'] !=0)
            
    
    if (negcount > poscount):
        sentiment = 'Negative'
    elif (poscount > negcount):
        sentiment = 'Positive'
    
    else:
        sentiment = 'Neutral'
    
    #
    return sentiment

    maindict = pd.read_excel('pathname to spreadsheet',
                        parse_cols=[0, 7,8,9,10,11,12,13])

maindict['Include'] = maindict.sum(axis=1)
smalldict = maindict.query('Include != 0',)
dictlist = smalldict['Word'].tolist()
print "Finance dictionary size: " + str(len(dictlist))

dictlist2 = map(lambda x: str(x).lower(), dictlist)