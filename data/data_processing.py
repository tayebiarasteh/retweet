"""
@authors:
Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
Mehrpad Monajem <mehrpad.monajem@fau.de>
"""

import numpy as np
import os
import pandas as pd
from configs.serde import *
import pdb

epsilon = 1e-15



def summarizer(data_path, input_file_name, output_file_name):
    data = pd.read_csv(os.path.join(data_path, input_file_name))
    data_new = data.drop(['user', 'reply'], axis=1)
    data_final = pd.DataFrame(columns=['label', 'id', 'tweet'])

    it = iter(range(0, len(data_new)))

    for tweet in it:

        #tweet: index
        label_pos = 0   # total number of positive labels for each tweet
        label_neg = 0
        label_neut = 0

        label = data_new.iloc[tweet][0]
        tweet_text = data_new.iloc[tweet][1]

        if label == 'positive':
            label_pos += 1
        elif label == 'negative':
            label_neg += 1
        elif label == 'neutral':
            label_neut += 1

        while True:
            tweet = next(it)
            if tweet_text == data_new.iloc[tweet][1]:
                if data_new.iloc[tweet][0] == 'positive':
                    label_pos += 1
                elif data_new.iloc[tweet][0] == 'negative':
                    label_neg += 1
                elif data_new.iloc[tweet][0] == 'neutral':
                    label_neut += 1
            else:
                break
            if tweet + 1 >= len(data_new):
                break
        var = {label_pos: "positive", label_neg: "negative", label_neut: "neutral"}

        # our proposed algorithm
        overall = label_pos + label_neg + label_neut
        if (label_neut/(overall + epsilon)) > 0.8:
            label_final = var.get(label_neut)
        else:
            if (label_pos/(label_neg + epsilon)) > 2:
                label_final = var.get(label_pos)
            elif (label_neg/(label_pos + epsilon)) > 2:
                label_final = var.get(label_neg)
            else:
                label_final = var.get(label_neut)

        df = pd.DataFrame([[label_final, data_new['id'][tweet], tweet_text]], columns=['label', 'id', 'tweet'])
        data_final = data_final.append(df)

    data_final.to_csv(os.path.join(data_path, output_file_name), index=False)



def reply_convertor():
    '''
    converting the individual test tweets with their replies to txt format in order to be easier to read manually
    outputs:
        :_org: is a file only contains the replies of a tweet
        :_to_be_filled: is a file with only one row and 2 columns, one is the tweet, the other is
         named CHANGE. You should replace this CHANGE with your final label and then save.
         Note: please remove the _org file after you replaced the CHANGE.
    '''
    path = "/home/soroosh/Documents/Repositories/twitter_sentiment/data/datasets/postreply/test_gold"
    output_path = "/home/soroosh/Documents/Repositories/twitter_sentiment/data/datasets/postreply/test_gold_out"
    file_list = [f for f in os.listdir(path) if f.endswith('.csv')]

    for idx, file in enumerate(file_list):
        data = pd.read_csv(os.path.join(path, file))
        data['reply'].to_csv(os.path.join(output_path, str(idx) + '_org.txt'), sep='\t')
        new_data = pd.DataFrame([['CHANGE', data['id'][0], data['tweet'][0]]], columns=['label', 'id', 'tweet'])
        new_data.to_csv(os.path.join(output_path, str(idx) + '_to_be_filled.txt'), sep='\t', index=False)


def manual_label_concat():
    '''
    concatenates the individual files from reply_convertor()
    '''
    path = "/home/soroosh/Documents/Repositories/twitter_sentiment/data/datasets/postreply/test_gold_out"
    file_list = [f for f in os.listdir(path) if f.endswith('.txt')]

    data_final = pd.DataFrame(columns=['label', 'id', 'tweet'])
    for file in file_list:
        data = pd.read_csv(os.path.join(path, file), sep='\t')
        data_final = data_final.append(data)
    data_final.to_csv(os.path.join(path, 'final_test_post_reply.csv'), index=False)



def post_reply_downloader(list_of_word, max_num_tweets, mode='download'):
    path = './preprocessing_utils/get_old_tweets_3-0.0.10'
    max_tweets = max_num_tweets
    list_word = list_of_word
    for word in list_word:
        print('The word is:', word,'- Max number of tweets:', max_tweets)
        os.system(' python %s/GetOldTweets3.py --querysearch "%s" --lang en --toptweets --maxtweets %s'
                  ' --output=%s/%s.csv' %(path, word, max_tweets, path, word))
    if mode == 'test':
        data = pd.read_csv('%s/%s.csv' % (path, list_word[0]))
        return data



if __name__=='__main__':
    summarizer(data_path="./datasets/postreply",
               input_file_name="data_post_reply_withlabel.csv",
               output_file_name="final_data_post_reply.csv")

    # pdb.set_trace()
    # a=2

