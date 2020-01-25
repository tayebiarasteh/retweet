
import pandas as pd

path = '/home/mehrpad/Desktop/dl_seminar/seminar_code'




# idx2word=dict(enumerate(sorted(set(" ".join([line.split("\t")[-1] for line in open('data/dataset_SemEval/message_level/train/2014_b_dev.txt').read().split("\n")]).split()))))
# dictionary=dict(enumerate(sorted(set(" ".join([line.split("\t")[-1] for line in open('data/dataset_SemEval/message_level/train/2014_b_dev.txt').read().split("\n")]).split()))))
# lables={'positive':0,"negative":1,"neutral":2}
# ds=[(d.split("\t")[-1].split(), lables[d.split("\t")[2]])  for d in data if len(d.split("\t"))>2]
# ds_test =[(d.split("\t")[-1].split(), lables[d.split("\t")[2]])  for d in data_test if len(d.split("\t"))>2]
# ds=[(d.split("\t")[-1].split(), d.split("\t")[2])  for d in data if len(d.split("\t"))>2]
# ds_test =[(d.split("\t")[-1].split(), d.split("\t")[2])  for d in data_test if len(d.split("\t"))>2]

data1 = (open(path + '/data/dataset_SemEval/message_level/train/2014_b_train.txt').read().split("\n"))
data1 = data1[:len(data1) - 1]
data2 = (open(path + '/data/dataset_SemEval/message_level/train/2014_b_dev.txt').read().split("\n"))
data2 = data2[:len(data2) - 1]
data = data1 + data2

data_test1 = (open(path + '/data/dataset_SemEval/message_level/test/2014_b_test_gold.txt').read().split("\n"))
data_test1 = data_test1[:len(data_test1) - 1]
data_test2 = (open(path + '/data/dataset_SemEval/message_level/test/2015_b_test_gold.txt').read().split("\n"))
data_test2 = data_test2[:len(data_test2) - 1]
data_test = data_test1 + data_test2

print(len(data))
print(len(data_test))

ds = [(d.split("\t")[0], d.split("\t")[1], d.split("\t")[2], d.split("\t")[3]) for d in data if
      len(d.split("\t")) > 2]
ds_test = [(d.split("\t")[0], d.split("\t")[1], d.split("\t")[2], d.split("\t")[3]) for d in data_test if
           len(d.split("\t")) > 2]

ds_dic = {'id': [seq[0] for seq in ds], 'user_id': [seq[1] for seq in ds],
          'label': [seq[2] for seq in ds], 'text': [seq[3] for seq in ds]}
ds_test_dic = {'id': [seq[0] for seq in ds_test], 'user_id': [seq[1] for seq in ds_test],
               'label': [seq[2] for seq in ds_test], 'text': [seq[3] for seq in ds_test]}

train = pd.DataFrame(ds_dic, columns=["id", "user_id", "text", "label"])
test = pd.DataFrame(ds_test_dic, columns=["id", "user_id", "text", "label"])

train.to_csv(path + "/nlp/data/train_SemEval.csv", index=False)
test.to_csv(path + "/nlp/data/test_SemEval.csv", index=False)


