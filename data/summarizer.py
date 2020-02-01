'''
"data_post_reply_withlabel.py" contains now the label of each reply, in the column 0.

#TODO [@Mehrpad or @Sundar]: for each unique tweet, count the number of positive, negative,
and neutral labels and pick the label which has the maximum
number of appearance and assign to the label of tweet. Then remove all the repetitive tweets.
(here we don't care about the replies and you can delete them if you want.

*** Our intended data structure here: Unique tweets and only one label for each tweet.
'''

import numpy as np

'''some sample code I was working on, which may be handy for you:'''

# positive_counter = 0
# negative_counter = 0
# neutral_counter = 0
#
# for idx, row in enumerate(data):
#     if idx != len(data) -1:
#         if data[idx][1] == data[idx + 1][1]:
#             if data[idx][1] == 'positive':
#                 positive_counter += 1
#             elif filenames[idx][1] == 'negative':
#                 negative_counter += 1
#             elif filenames[idx][1] == 'neutral':
#                 neutral_counter += 1
#         else:
#             if data[idx][1] == 'positive':
#                 positive_counter += 1
#             elif data[idx][1] == 'negative':
#                 negative_counter += 1
#             elif data[idx][1] == 'neutral':
#                 neutral_counter += 1
#             if max(positive_counter, negative_counter, neutral_counter) == positive_counter:
#                 label = 'positive'
#             elif max(positive_counter, negative_counter, neutral_counter) == negative_counter:
#                 label = 'negative'
#             else:
#                 label = 'neutral'
