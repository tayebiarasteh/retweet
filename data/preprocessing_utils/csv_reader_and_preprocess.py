"""
Read all csv files with post_reply_downloader.py file and concat them.
Also it drops the column that is not necessary for the task.

@author: Mehrpad Monajem <mehrpad.monajem@fau.de>
"""

import pandas as pd
import glob


path = './data/preprocessing_utils/GetOldTweets3-0.0.10'
path_new = path + '/post_reply'
print(path_new)
list_files = glob.glob('./*.csv', recursive=True)
print(list_files)
print(len(list_files))


i = 0
for name in list_files:

    data = pd.read_csv('%s' % name)
    if i == 0:
        data_new = data
    else:
        data_new = pd.concat([data_new, data], ignore_index=True)
    i += 1
    print(i)

print('finish reading data')
print(len(data_new))
data_new.to_csv("./new/data.csv")
data_new = data_new.drop(['replies', 'retweets', 'link', 'reply_id', 'reply_username'], axis=1)
print(data_new)
data_final = pd.DataFrame(columns = ['tweet', 'id', 'user', 'reply'])

data_new.to_csv("./new/post_reply.csv")
data_final['tweet'] = data_new['text']
data_final['id'] = data_new['id']
data_final['user'] = data_new['username']
data_final['reply'] = data_new['reply_text']

reply = data_final['reply']
reply_new = []
for r in reply:
    zz = list(filter(lambda word: word[0]!='@', r.split()))
    zz = " ".join(zz)
    reply_new.append(zz)


data_final['reply'] = reply_new

index_empty_row = data_final[data_final['reply'] == ''].index
index_empty_row = pd.Index.tolist(index_empty_row)



data_final = data_final.drop(data_final.index[index_empty_row])

# Delete row at index position 0 & 1
data_final.to_csv("./new/data_post_reply.csv")
