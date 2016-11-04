def twitterAuth(consumer_key, consumer_secret):
    auth = tweepy.OAuthHandler(consumer_key    = consumer_key,
                               consumer_secret =  consumer_secret)
    api = tweepy.API(auth)
    return api 


# api = twitterAuth(consumer_key, consumer_secret)

def twitterSearchTerm(api, term):
    q = term
    result = api.search(q=q)
    len(result)

    results = []
    for tweet in tweepy.Cursor(api.search, q=q).items(500):
        results.append(tweet)
        
    return results 


    def toDataFrame(tweets):
    DataSet = pd.DataFrame()
    
    DataSet['tweetID']     = [tweet.id for tweet in tweets]
    DataSet['source']      = [tweet.source for tweet in tweets]
    DataSet['sourceURL']   = [tweet.source_url for tweet in tweets]
    
    DataSet['userID']      = [tweet.user.id for tweet in tweets]
    DataSet['userScreen']  = [tweet.user.screen_name for tweet in tweets]
    DataSet['userName']    = [tweet.user.name for tweet in tweets]
    
    DataSet['userCreateDt']    = [tweet.user.created_at for tweet in tweets]
    DataSet['userDesc']        = [tweet.user.description for tweet in tweets]
    DataSet['userFollowerCt']  = [tweet.user.followers_count for tweet in tweets]
    DataSet['userFriendsCt']   = [tweet.user.friends_count for tweet in tweets]
    DataSet['userLocation']    = [tweet.user.location for tweet in tweets]
    DataSet['userCoordinates'] = [tweet.coordinates for tweet in tweets]
    DataSet['userTimezone']    = [tweet.user.time_zone for tweet in tweets]  
    
    DataSet['tweetText']       = [tweet.text for tweet in tweets]
    DataSet['tweetRetweetCt']  = [tweet.retweet_count for tweet in tweets]
    DataSet['tweetFavoriteCt'] = [tweet.favorite_count for tweet in tweets]
    DataSet['tweetSource']     = [tweet.source for tweet in tweets]
    DataSet['tweetCreated']    = [tweet.created_at for tweet in tweets]
    
    return DataSet

#DataSet = toDataFrame(results)
#DataSet.to_csv('DataSet.csv', sep=',')
