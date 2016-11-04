
##########  Beginning CountVectorizer  ########## 

from sklearn.feature_extraction.text import CountVectorizer
CountV2 = CountVectorizer(binary        = True, 
                          lowercase     = True, 
                          stop_words    = 'english', 
                          strip_accents = 'ascii',
                          max_df        = .90, 
                          min_df        = .0003,
                          max_features  = None)

CountV1_dm    = CountV1.fit_transform(DATA)


shapeList = [CountV1_dm.shape,
		CountV2_dm.shape]

dfCountV1 = pd.DataFrame(CountV1_dm.toarray(), 
             columns = CountV1.get_feature_names())

V1vV2 = [i for i, j in zip(dfCountV1, dfCountV2) if i != j]
##########  Ending CountVectorizer  ########## 


##########  Beginning TfidfVectorizer  ########## 

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf1 = TfidfVectorizer(analyzer      = 'word',
                         binary        = True,
                         decode_error  = 'strict',
                         encoding      = 'utf-8',
                         lowercase     = True,
                         max_df        = .93, 
                         min_df        = .0104,
                         max_features  = None,
                         ngram_range   = (2, 2),
                         norm          = 'l2',
                         preprocessor  = None,
                         smooth_idf    = True,
                         stop_words    = 'english',
                         sublinear_tf  = False,
                         token_pattern = '(?u)\\b\\w\\w+\\b',
                         tokenizer     = None,
                         use_idf       = True,
                         vocabulary    = None)

tf1_dm = tfidf1.fit_transform(filtered)

shapeList3 = [tf1_dm.shape,             
		tf2_dm.shape ]

df_Tfidf1V1 = pd.DataFrame(tfidf1_dm.toarray(), 
             columns = tfidf1.get_feature_names())

V1vV2 = [i for i, j in zip(df_Tfidf1V1, df_Tfidf1V2) if i != j]
##########  End TfidfVectorizer  ########## 
