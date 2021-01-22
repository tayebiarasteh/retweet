"""
Functions for data pre-processing, downloading.

@authors:
Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
Mehrpad Monajem <mehrpad.monajem@fau.de>
"""

import numpy as np
import os
import pandas as pd
from configs.serde import *
import pdb
import time
from main import *
from tqdm import tqdm

epsilon = 1e-15



def summarizer(data_path, input_file_name, output_file_name):
    '''
    Chooses a final label for each tweet from the list of reply-labels it gets
    :param data_path: directory where the data located
    :param input_file_name: name of the input file (data_post_reply_withlabel.csv)
    :param output_file_name: name of the input file (final_data_post_reply.csv)
    '''
    start_time = time.time()

    data = pd.read_csv(os.path.join(data_path, input_file_name))
    data_new = data.drop(['user', 'reply'], axis=1)
    data_final = pd.DataFrame(columns=['label', 'id', 'tweet'])

    it = iter(range(0, len(data_new)))

    for tweet in tqdm(it):

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
        if (label_neut/(overall + epsilon)) > 0.85:
            label_final = var.get(label_neut)
        else:
            if (label_pos/(label_neg + epsilon)) > 1.5:
                label_final = var.get(label_pos)
            elif (label_neg/(label_pos + epsilon)) > 1.6:
                label_final = var.get(label_neg)
            else:
                label_final = var.get(label_neut)

        df = pd.DataFrame([[label_final, data_new['id'][tweet], tweet_text]], columns=['label', 'id', 'tweet'])
        data_final = data_final.append(df)
    data_final = data_final.sample(frac=1)
    data_final.to_csv(os.path.join(data_path, output_file_name), index=False)
    # Duration
    end_time = time.time()
    test_mins, test_secs = prediction_time(start_time, end_time)
    print(f'Total Summarizer Time: {test_mins}m {test_secs}s')


def reply_convertor():
    '''
    converting the individual test tweets with their replies to txt format in order to be easier to read manually
    outputs:
        :_org: is a file only contains the replies of a tweet
        :_to_be_filled: is a file with only one row and 2 columns, one is the tweet, the other is
         named CHANGE. You should replace this CHANGE with your final label and then save.
         Note: please remove the _org file after you replaced the CHANGE.
    '''
    path = './data/datasets/' \
           'postreply/Gold set/Group_4_4431/Unlabeled_g4/Group4_4431_unlabeled/'
    output_path = './data/datasets/' \
                  'postreply/Gold set/Group_4_4431/Unlabeled_g4/Group4_4431_out_unlabeled/'
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
    path = "./data/datasets/" \
           "postreply/Gold set/Group_3_3931/Labeled_g3/Hamid_Group3_3931"
    file_list = [f for f in os.listdir(path) if f.endswith('.txt')]

    data_final = pd.DataFrame(columns=['label', 'id', 'tweet'])
    for file in file_list:
        data = pd.read_csv(os.path.join(path, file), sep='\t')
        if data.columns[0] == 'label' and data.columns[1] == 'id' and data.columns[2] == 'tweet':
            if data['label'][0] == 'neutral' or data['label'][0] == 'positive' or data['label'][0] == 'negative':
                data_final = data_final.append(data)
    data_final = data_final.sample(frac=1)
    data_final.to_csv(os.path.join(path, 'final_test_post_reply.csv'), index=False)


def tweet_correlator():
    '''
    Checks whether two manual annotators have the same opinion on each tweet and saves only when they aggree.
    '''
    path_hamid = "./data/" \
                 "datasets/postreply/Gold set/Group_3_3931/Labeled_g3/Hamid_Group3_3931/final_test_post_reply.csv"
    path_mahshad = "./data/" \
                   "datasets/postreply/Gold set/Group_3_3931/Labeled_g3/Mahshad_Group3_3931/final_test_post_reply.csv"
    data2 = pd.read_csv(path_hamid)
    data = pd.read_csv(path_mahshad)
    final_data = pd.DataFrame(columns=['label', 'id', 'tweet'])

    for idx, ID in enumerate(data['id']):
        for idx2, ID2 in enumerate(data2['id']):
            if ID == ID2:
                if data['label'][idx] == data2['label'][idx2]:
                    df = pd.DataFrame([[data2['label'][idx2], data2['id'][idx2], data2['tweet'][idx2]]],
                                      columns=['label', 'id', 'tweet'])
                    final_data = final_data.append(df)
                    final_data= final_data.sample(frac=1)
    final_data.to_csv("./"
                      "data/datasets/postreply/Gold set/Group_3_3931/Labeled_g3/"
                      "Correlated_Group3_3931/final_test_post_reply.csv", index=False)

    # the uncorrelated ones
    path_g1 = "./data/" \
                   "datasets/postreply/Gold set/Group_3_3931/Unlabeled_g3/Group3_3931_unlabeled"
    path_g1_corr = "./data/" \
                   "datasets/postreply/Gold set/Group_3_3931/Labeled_g3/Correlated_Group3_3931/final_test_post_reply.csv"
    path_g1_uncorr = "./data/" \
                   "datasets/postreply/Gold set/Group_3_3931/Labeled_g3/Uncorrelated"

    data = pd.read_csv(path_g1_corr)

    file_list = [f for f in os.listdir(path_g1) if f.endswith('.csv')]
    for idx, file in enumerate(file_list):
        data_file = pd.read_csv(os.path.join(path_g1, file))
        if (data_file['id'][0] == data['id']).sum() == 0:
            data_file['reply'].to_csv(os.path.join(path_g1_uncorr, str(idx) + '_org.txt'), sep='\t')
            new_data = pd.DataFrame([['CHANGE', data['id'][0], data['tweet'][0]]], columns=['label', 'id', 'tweet'])
            new_data.to_csv(os.path.join(path_g1_uncorr, str(idx) + '_to_be_filled.txt'), sep='\t', index=False)


def correlated_tweet_balancing():
    '''
    balancing the classes of the correlated test data
    '''
    path2 = "./data/datasets/" \
            "postreply/Gold set/Group_2_3177/Labeled_g2/Correlated_Group2_3177/final_test_post_reply.csv"
    path1 = "./data/datasets/" \
              "postreply/Gold set/Group_1_968/Labeled_g1/Correlated_Group1_968/final_test_post_reply.csv"
    path3 = "./data/datasets/" \
              "postreply/Gold set/Group_3_3931/Labeled_g3/Correlated_Group3_3931/final_test_post_reply.csv"
    path_final1 = "./data/datasets/" \
              "postreply/Gold set/Group_1_968/Labeled_g1/Correlated_Group1_968/final_test_post_reply_balanced.csv"
    path_final2 = "./data/datasets/" \
                 "postreply/Gold set/Group_2_3177/Labeled_g2/Correlated_Group2_3177/final_test_post_reply_balanced.csv"
    path_final3 = "./data/datasets/" \
              "postreply/Gold set/Group_3_3931/Labeled_g3/Correlated_Group3_3931/final_test_post_reply_balanced.csv"
    data = pd.read_csv(path1)

    counter_neu = 0
    counter_neg = 0
    counter_pos = 0
    neu_list = []
    neg_list = []
    pos_list = []

    for idx, item in enumerate(data['label']):
        if counter_neu < 100:
            if item == 'neutral':
                neu_list.append(idx)
                counter_neu += 1
        #         elif counter_neg < 100:
        #             if item == 'negative':
        #                 neg_list.append(idx)
        #                 counter_neg += 1
        elif counter_pos < 100:
            if item == 'positive':
                pos_list.append(idx)
                counter_pos += 1

    data = data.drop(neu_list)
    #     data = data.drop(neg_list)
    data = data.drop(pos_list)
    data.to_csv(path_final1, index=False)


def gold_data_concat():
    '''
    concatenating the final gold dataset from each group to make the final test data
    '''
    path_g2 = "./data/datasets/" \
                 "postreply/Gold set/Group_2_3177/Labeled_g2/Correlated_Group2_3177/final_test_post_reply_balanced.csv"
    path_g1 = "./data/datasets/" \
              "postreply/Gold set/Group_1_968/Labeled_g1/Correlated_Group1_968/final_test_post_reply_balanced.csv"
    path_g3 = "./data/datasets/" \
              "postreply/Gold set/Group_3_3931/Labeled_g3/Correlated_Group3_3931/final_test_post_reply_balanced.csv"
    path_g1_soroosh = "./data/datasets/" \
              "postreply/Gold set/Group_1_968/Labeled_g1/Soroosh_Group1_968/final_test_post_reply.csv"
    path_final = "./data/datasets/" \
              "postreply/final_test_post_reply.csv"

    data = pd.read_csv(path_g1)
    data3 = pd.read_csv(path_g3)
    data2 = pd.read_csv(path_g2)
    data2 = data2.append(data)
    data2 = data2.append(data3)
    data2 = data2.sample(frac=1)
    data2.to_csv(path_final, index=False)


def philipp_getoldtweet_concat():
    getold = "./data/datasets/" \
             "postreply/final_data_post_reply.csv"
    philipp = "./data/datasets/" \
              "postreply/philipp_final.csv"
    output = "./data/datasets/" \
             "postreply/training_data_post_reply.csv"

    data = pd.read_csv(getold)
    data2 = pd.read_csv(philipp)
    data = data.append(data2)
    data = data.sample(frac=1)
    data.to_csv(output, index=False)


def counting_pie_chart():
    '''
    For the visualization of the distribution of the classes in the train and test sets
    '''
    training = './' \
               'data/datasets/postreply/training_data_post_reply.csv'
    test = './' \
           'data/datasets/postreply/final_test_post_reply.csv'

    data = pd.read_csv(training)
    positive_count = (data['label'] == 'positive').sum()
    negative_count = (data['label'] == 'negative').sum()
    neutral_count = (data['label'] == 'neutral').sum()
    total = negative_count + positive_count + neutral_count
    print('train set')
    print('positive:', positive_count/total)
    print('negative:', negative_count/total)
    print('neutral:', neutral_count/total)

    data = pd.read_csv(test)
    positive_count = (data['label'] == 'positive').sum()
    negative_count = (data['label'] == 'negative').sum()
    neutral_count = (data['label'] == 'neutral').sum()
    total = negative_count + positive_count + neutral_count
    print('\ntest set')
    print('positive:', positive_count/total)
    print('negative:', negative_count/total)
    print('neutral:', neutral_count/total)


def test_from_train_creator():
    '''
    Reduces 1000 tweets from the training data and transforms them to the test form.
    '''
    output_path = './data/datasets/' \
                  'postreply/data_post_reply_withlabel.csv'
    output_path2 = './data/datasets/' \
                   'postreply/Gold set/Group_4_4431/Unlabeled_g4/Group4_4431_unlabeled/'
    output_path3 = './data/datasets/' \
                   'data_post_reply_withlabel.csv'

    data = pd.read_csv(output_path)
    trainingdatagetold = pd.DataFrame(columns=['label', 'tweet', 'id', 'user', 'reply'])

    for idx, item in enumerate(set(data['id'])):
        print(idx)
        if idx < 500:
            testgetold = pd.DataFrame(columns=['label', 'tweet', 'id', 'user', 'reply'])
            testgetold = testgetold.append(data[data['id'] == item])
            testgetold.to_csv(os.path.join(output_path2, str(idx) + '.csv'), index=False)

        else:
            trainingdatagetold = trainingdatagetold.append(data[data['id'] == item])

    trainingdatagetold.to_csv(output_path3, index=False)



def post_reply_downloader(list_of_word, max_num_tweets, mode='download'):
    '''
    Manual tweet and reply downloader from Twitter
    :param list_of_word: list of keywords to search for
    :param max_num_tweets:
    :param mode: 'download' or 'test'
    '''
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
    # summarizer("./datasets/postreply", "philipp_withlabel.csv", "philipp_final.csv")
    # summarizer("./datasets/postreply", "data_post_reply_withlabel.csv", "final_data_post_reply.csv")
    # summarizer("./datasets/postreply/Gold set", "correlated_balanced_tweetandreply_labeled.csv", "correlated_balanced_tweetandreply_final.csv")
    # philipp_getoldtweet_concat()
    counting_pie_chart()
    # manual_label_concat()
    # tweet_correlator()
    # test_from_train_creator()
    # reply_convertor()
    # gold_data_concat()
    # correlated_tweet_balancing()
