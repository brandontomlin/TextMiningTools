locs = [('Omnicom', 'IN', 'New York'),
('DDB Needham', 'IN', 'New York'),
('Kaplan Thaler Group', 'IN', 'New York'),
('BBDO South', 'IN', 'Atlanta'),
('Georgia-Pacific', 'IN', 'Atlanta')]

query = [e1 for (e1, rel, e2) in locs if e2=='Atlanta']
print(query)

def ie_preprocess(document):
	sentences = nltk.sent_tokenize(document)
	sentences = [nltk.word_tokenize(sent) for sent in sentences]
	sentences = [nltk.pos_tag(sent) for sent in sentences]

# 2.1   Noun Phrase Chunking




sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
	("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]

grammar = "NP: {<DT>?<JJ>*<NN>}" 

cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)

# 2.3   Chunking with Regular Expressions

grammar = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""
cp = nltk.RegexpParser(grammar)
sentence = [("Rapunzel", "NNP"), ("let", "VBD"), ("down", "RP"), [1]
                 ("her", "PP$"), ("long", "JJ"), ("golden", "JJ"), ("hair", "NN")]

print(cp.parse(sentence))
nouns = [("money", "NN"), ("market", "NN"), ("fund", "NN")]
grammar = "NP: {<NN><NN>}  # Chunk two consecutive nouns"
cp = nltk.RegexpParser(grammar)
print(cp.parse(nouns))

# 2.4   Exploring Text Corpora



cp = nltk.RegexpParser('CHUNK: {<V.*> <TO> <V.*>}')
brown = nltk.corpus.brown
for sent in brown.tagged_sents():
	tree = cp.parse(sent)
	for subtree in tree.subtrees():
		if subtree.label() == 'CHUNK': print(subtree)

# 2.5   Chinking

grammar = r"""
  NP:
    {<.*>+}               # Chunk everything
    }<VBD|IN>+{      # Chink sequences of VBD and IN
  """
sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
       ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
cp = nltk.RegexpParser(grammar)


print(cp.parse(sentence))

# Representing Chunks: Tags vs Trees


# As befits their intermediate status between tagging and parsing (8.), chunk structures can be 
# represented using either tags or trees. The most widespread file representation uses IOB tags. 
# In this scheme, each token is tagged with one of three special chunk tags, I (inside), O (outside), 
# or B (begin). A token is tagged as B if it marks the beginning of a chunk. Subsequent tokens within 
# the chunk are tagged I. All other tokens are tagged O. The B and I tags are suffixed with the chunk 
# type, e.g. B-NP, I-NP. Of course, it is not necessary to specify a chunk type for tokens that appear
#  outside a chunk, so these are just labeled O. An example of this scheme is shown in 2.5.

# ../images/chunk-tagrep.png
# Figure 2.5: Tag Representation of Chunk Structures
# IOB tags have become the standard way to represent chunk structures in files, and we will also be 
# using this format. Here is how the information in 2.5 would appear in a file:

# We PRP B-NP
# saw VBD O
# the DT B-NP
# yellow JJ I-NP
# dog NN I-NP
# In this representation there is one token per line, each with its part-of-speech tag and chunk tag. 
# This format permits us to represent more than one chunk type, so long as the chunks do not overlap.
#  As we saw earlier, chunk structures can also be represented using trees. These have the benefit that 
#  each chunk is a constituent that can be manipulated directly. An example is shown in 2.6.

# ../images/chunk-treerep.png
# Figure 2.6: Tree Representation of Chunk Structures
# Note

# NLTK uses trees for its internal representation of chunks, but provides methods for reading and 
# writing such trees to the IOB format.






# Developing and Evaluating Chunkers

# Now you have a taste of what chunking does, but we haven't explained how to evaluate chunkers. 
# As usual, this requires a suitably annotated corpus. We begin by looking at the mechanics of converting 
# IOB format into an NLTK tree, then at how this is done on a larger scale using a chunked corpus. 
# We will see how to score the accuracy of a chunker relative to a corpus, then look at some more 
# data-driven ways to search for NP chunks. Our focus throughout will be on expanding the coverage 
# of a chunker



# Reading IOB Format and the CoNLL 2000 Corpus

# Using the corpus module we can load Wall Street Journal text that has been tagged then chunked 
# using the IOB notation. The chunk categories provided in this corpus are NP, VP and PP. 
# As we have seen, each sentence is represented using multiple lines, as shown below:

# he PRP B-NP
# accepted VBD B-VP
# the DT B-NP
# position NN I-NP
# ...
# A conversion function chunk.conllstr2tree() builds a tree representation from 
# one of these multi-line strings. Moreover, it permits us to choose any subset of the three chunk types to use, 
# here just for NP chunks:

text = '''
he PRP B-NP
accepted VBD B-VP
the DT B-NP
position NN I-NP
of IN B-PP
vice NN B-NP
chairman NN I-NP
of IN B-PP
Carlyle NNP B-NP
Group NNP I-NP
, , O
a DT B-NP
merchant NN I-NP
banking NN I-NP
concern NN I-NP
. . O
'''
nltk.chunk.conllstr2tree(text, chunk_types=['NP']).draw()

from nltk.corpus import conll2000
print(conll2000.chunked_sents('train.txt')[99])

print(conll2000.chunked_sents('train.txt', chunk_types=['NP'])[99])

from nltk.corpus import conll2000
cp = nltk.RegexpParser("")
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
print(cp.evaluate(test_sents))


grammar = r"NP: {<[CDJNP].*>+}"
cp = nltk.RegexpParser(grammar)
print(cp.evaluate(test_sents))

# As you can see, this approach achieves decent results. However, we can improve on it by adopting a 
# more data-driven approach, where we use the training corpus to find the chunk tag (I, O, or B) that is 
# most likely for each part-of-speech tag. In other words, we can build a chunker using a unigram tagger 
# (4). But rather than trying to determine the correct part-of-speech tag for each word, we are trying to 
# determine the correct chunk tag, given each word's part-of-speech tag.

# In 3.1, we define the UnigramChunker class, which uses a unigram tagger to label sentences with chunk tags. 
# Most of the code in this class is simply used to convert back and forth between the chunk tree representation 
# used by NLTK's ChunkParserI interface, and the IOB representation used by the embedded tagger. 
# The class defines two methods: a constructor [1] which is called when we build a new UnigramChunker; and 
# the parse method [3] which is used to chunk new sentences.

class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
unigram_chunker = UnigramChunker(train_sents)
print(unigram_chunker.evaluate(test_sents))

postags = sorted(set(pos for sent in train_sents
                      for (word,pos) in sent.leaves()))
print(unigram_chunker.tagger.tag(postags))


bigram_chunker = BigramChunker(train_sents)
print(bigram_chunker.evaluate(test_sents))


# 3.3   Training Classifier-Based Chunkers


# These two sentences have the same part-of-speech tags, yet they are chunked differently. In the first sentence, the farmer and rice are separate chunks, while the corresponding material in the second sentence, the computer monitor, is a single chunk. Clearly, we need to make use of information about the content of the words, in addition to just their part-of-speech tags, if we wish to maximize chunking performance.

# One way that we can incorporate information about the content of words is to use a classifier-based tagger to chunk the sentence. Like the n-gram chunker considered in the previous section, this classifier-based chunker will work by assigning IOB tags to the words in a sentence, and then converting those tags to chunks. For the classifier-based tagger itself, we will use the same approach that we used in 1 to build a part-of-speech tagger.

# The basic code for the classifier-based NP chunker is shown in 3.2. It consists of two classes. The first class [1] is almost identical to the ConsecutivePosTagger class from 1.5. The only two differences are that it calls a different feature extractor [2] and that it uses a MaxentClassifier rather than a NaiveBayesClassifier [3]. The second class [4] is basically a wrapper around the tagger class that turns it into a chunker. During training, this second class maps the chunk trees in the training corpus into tag sequences; in the parse() method, it converts the tag sequence provided by the tagger back into a chunk tree.



class ConsecutiveNPChunkTagger(nltk.TaggerI): 

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train( 
            train_set, algorithm='megam', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

def npchunk_features(sentence, i, history):
	word, pos = sentence[i]
	return {"pos": pos}

chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))

def npchunk_features(sentence, i, history):
	word, pos = sentence[i]
	if i == 0:
	 	prevword, prevpos = "<START>", "<START>"
	else:
	 	prevword, prevpos = sentence[i-1]
	return {"pos": pos, "prevpos": prevpos}

chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))


def npchunk_features(sentence, i, history):
	word, pos = sentence[i]
	if i == 0:
		prevword, prevpos = "<START>", "<START>"
	else:
		prevword, prevpos = sentence[i-1]
	return {"pos": pos, "word": word, "prevpos": prevpos}

chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))


def npchunk_features(sentence, i, history):
word, pos = sentence[i]
     if i == 0:
         prevword, prevpos = "<START>", "<START>"
     else:
         prevword, prevpos = sentence[i-1]
     if i == len(sentence)-1:
         nextword, nextpos = "<END>", "<END>"
     else:
         nextword, nextpos = sentence[i+1]
     return {"pos": pos,
             "word": word,
             "prevpos": prevpos,
             "nextpos": nextpos, [1]
             "prevpos+pos": "%s+%s" % (prevpos, pos), 
             "pos+nextpos": "%s+%s" % (pos, nextpos),
             "tags-since-dt": tags_since_dt(sentence, i)} 

 	
def tags_since_dt(sentence, i):
     tags = set()
     for word, pos in sentence[:i]:
         if pos == 'DT':
             tags = set()
         else:
             tags.add(pos)
     return '+'.join(sorted(tags))

chunker = ConsecutiveNPChunker(train_sents)
print(chunker.evaluate(test_sents))



# Finally, we can try extending the feature extractor with a variety of additional features, such as lookahead features [1], paired features [2], and complex contextual features [3]. This last feature, called tags-since-dt, creates a string describing the set of all part-of-speech tags that have been encountered since the most recent determiner, or since the beginning of the sentence if there is no determiner before index i. .


# 4   Recursion in Linguistic Structure

# Building Nested Structure with Cascaded Chunkers




grammar = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
  """
cp = nltk.RegexpParser(grammar)
sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
    ("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]

    print(cp.parse(sentence))


 sentence = [("John", "NNP"), ("thinks", "VBZ"), ("Mary", "NN"),
     ("saw", "VBD"), ("the", "DT"), ("cat", "NN"), ("sit", "VB"),
     ("on", "IN"), ("the", "DT"), ("mat", "NN")]
print(cp.parse(sentence))

cp = nltk.RegexpParser(grammar, loop=2)
print(cp.parse(sentence))


# Trees


tree1 = nltk.Tree('NP', ['Alice'])
print(tree1)

tree2 = nltk.Tree('NP', ['the', 'rabbit'])
print(tree2)

tree3 = nltk.Tree('VP', ['chased', tree2])
tree4 = nltk.Tree('S', [tree1, tree3])
print(tree4)

print(tree4[1])

tree4[1].label()
tree4.leaves()
tree4[1][1][1]

tree3.draw() 

def traverse(t):
    try:
        t.label()
    except AttributeError:
        print(t, end=" ")
    else:
        # Now we know that t.node is defined
        print('(', t.label(), end=" ")
        for child in t:
            traverse(child)
        print(')', end=" ")

t = nltk.Tree('(S (NP Alice) (VP chased (NP the rabbit)))')
traverse(t)

# Note

# We have used a technique called duck typing to detect that t is a tree (i.e. t.label() is defined).



# 5   Named Entity Recognition

sent = nltk.corpus.treebank.tagged_sents()[22]
print(nltk.ne_chunk(sent, binary=True))
print(nltk.ne_chunk(sent)) 


IN = re.compile(r'.*\bin\b(?!\b.+ing)')
for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
     for rel in nltk.sem.extract_rels('ORG', 'LOC', doc,
                                      corpus='ieer', pattern = IN):
         print(nltk.sem.rtuple(rel))

from nltk.corpus import conll2002

vnv = """
 (
 is/V|    # 3rd sing present and
 was/V|   # past forms of the verb zijn ('be')
 werd/V|  # and also present
 wordt/V  # past of worden ('become)
 )
 .*       # followed by anything
 van/Prep # followed by van ('of')
 """
VAN = re.compile(vnv, re.VERBOSE)
for doc in conll2002.chunked_sents('ned.train'):
     for r in nltk.sem.extract_rels('PER', 'ORG', doc,
                                    corpus='conll2002', pattern=VAN):
        print(nltk.sem.clause(r, relsym="VAN")) [1]




# Note

# Your Turn: Replace the last line [1], by print(rtuple(rel, lcon=True, rcon=True)). This will show you the actual words that intervene between the two NEs and also their left and right context, within a default 10-word window. With the help of a Dutch dictionary, you might be able to figure out why the result VAN('annie_lennox', 'eurythmics') is a false hit.