'''
post reply downloader

@author: Mehrpad Monajem <mehrpad.monajem@fau.de>
'''

import os
import pandas as pd

path = '/home/mehrpad/Desktop/dl_seminar/git_seminar/data/get_old_tweets_3-0.0.10'

def post_reply_downloader(list_of_word, max_num_tweets, mode='download'):
    max_tweets = max_num_tweets
    list_word = list_of_word

    for word in list_word:
        print('The word is:', word,'- Max number of tweets:', max_tweets)
        os.system(' python %s/GetOldTweets3.py --querysearch "%s" --lang en --toptweets --maxtweets %s'
                  ' --output=%s/%s.csv' %(path, word, max_tweets, path, word))

    if mode == 'test':
        data = pd.read_csv('%s/%s.csv' % (path, list_word[0]))

        return data



