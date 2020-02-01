'''
"data_post_reply_withlabel.py" contains now the label of each reply, in the column 0.
and neutral labels and pick the label which has the maximum
number of appearance and assign to the label of tweet. Then remove all the repetitive tweets.
(here we don't care about the replies and you can delete them if you want.

*** Our intended data structure here: Unique tweets and only one label for each tweet.
@author: Mehrpad Monajem <mehrpad.monajem@fau.de>
'''

import pandas as pd

path = '/home/mehrpad/Desktop/dl_seminar/git_seminar/data'
path_new = path + '/post_reply'

data = pd.read_csv(path + '/data_post_reply_withlabel.csv')

print('finish reading data')
print(len(data))

data_new = data.drop(['id', 'user', 'reply'], axis=1)
data_final = pd.DataFrame(columns=['label', 'tweet'])

it = iter(range(0, len(data_new)))

for tweet in it:

    id = tweet
    label_pos = 0
    label_neg = 0
    label_nat = 0
    label = data_new.iloc[tweet][0]
    tweet_text = data_new.iloc[tweet][1]

    if label == 'positive':
        label_pos += 1
    elif label == 'negative':
        label_neg += 1
    elif label == 'neutral':
        label_nat += 1

    while True:
        tweet = next(it)
        if tweet_text == data_new.iloc[tweet][1]:
            if data_new.iloc[tweet][0] == 'positive':
                label_pos += 1
            elif data_new.iloc[tweet][0] == 'negative':
                label_neg += 1
            elif data_new.iloc[tweet][0] == 'neutral':
                label_nat += 1

        else:
            break
        if tweet + 1 >= len(data_new):
            break

    var = {label_pos: "positive", label_neg: "negative", label_nat: "neutral"}
    label_final = var.get(max(var))

    df = pd.DataFrame([[label_final, tweet_text]], columns=['label', 'tweet'])
    data_final = data_final.append(df)

data_final.to_csv(path + "/data_post_with_max_replies_label.csv", index=False)



