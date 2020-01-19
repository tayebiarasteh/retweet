import torch
from torchtext import data
import time
from torchtext.data import Field, Dataset, BucketIterator, TabularDataset
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext import datasets
import torchtext
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = '/home/mehrpad/Desktop/dl_seminar/seminar_code'
SEED = 1234
batch_size = 64
num_epoch = 15

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

class RNN(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))
    
def data_load():

    # idx2word=dict(enumerate(sorted(set(" ".join([line.split("\t")[-1] for line in open('data/dataset_SemEval/message_level/train/2014_b_dev.txt').read().split("\n")]).split()))))
    # dictionary=dict(enumerate(sorted(set(" ".join([line.split("\t")[-1] for line in open('data/dataset_SemEval/message_level/train/2014_b_dev.txt').read().split("\n")]).split()))))
    # lables={'positive':0,"negative":1,"neutral":2}
    # ds=[(d.split("\t")[-1].split(), lables[d.split("\t")[2]])  for d in data if len(d.split("\t"))>2]
    # ds_test =[(d.split("\t")[-1].split(), lables[d.split("\t")[2]])  for d in data_test if len(d.split("\t"))>2]
    # ds=[(d.split("\t")[-1].split(), d.split("\t")[2])  for d in data if len(d.split("\t"))>2]
    # ds_test =[(d.split("\t")[-1].split(), d.split("\t")[2])  for d in data_test if len(d.split("\t"))>2]

    data = (open(path + '/data/dataset_SemEval/message_level/train/2014_b_train.txt').read().split("\n"))
    data = data[:len(data)-1]
    data_test = (open(path + '/data/dataset_SemEval/message_level/train/2014_b_dev.txt').read().split("\n"))
    data_test = data[:len(data_test)-1]

    ds=[(d.split("\t")[0], d.split("\t")[1], d.split("\t")[2], d.split("\t")[3])  for d in data if len(d.split("\t"))>2]
    ds_test=[(d.split("\t")[0], d.split("\t")[1], d.split("\t")[2], d.split("\t")[3])  for d in data_test if len(d.split("\t"))>2]

    ds_dic = {'id': [seq[0] for seq in ds], 'user_id': [seq[1] for seq in ds],
              'label' : [seq[2] for seq in ds], 'text': [seq[3] for seq in ds]}
    ds_test_dic = {'id': [seq[0] for seq in ds_test], 'user_id': [seq[1] for seq in ds_test],
              'label' : [seq[2] for seq in ds_test], 'text': [seq[3] for seq in ds_test]}

    train_total = pd.DataFrame(ds_dic, columns=["id", "user_id", "text", "label"])
    train, val = train_test_split(train_total, test_size=0.2)
    test = pd.DataFrame(ds_test_dic, columns=["id", "user_id", "text", "label"])

    train.to_csv(path + "/data/dataset_SemEval/message_level/pre_data/train.csv", index=False)
    val.to_csv(path + "/data/dataset_SemEval/message_level/pre_data/val.csv", index=False)
    test.to_csv(path + "/data/dataset_SemEval/message_level/pre_data/test.csv", index=False)

    fields = [('id', None), ('user_id', None), ('text', TEXT), ('label', LABEL)]

    train_dataset, val_dataset = TabularDataset.splits(path= path + '/data/dataset_SemEval/message_level/pre_data', train='train.csv',
                                    validation='val.csv', format='csv', skip_header=True, fields=fields)
    test_dataset = TabularDataset(path=path + '/data/dataset_SemEval/message_level/pre_data/test.csv',
                                    format='csv', skip_header=True, fields=fields)

    print(train_dataset.examples[0])
    print(type(train_dataset.examples[0]))
    print(vars(train_dataset.examples[0]))
    print(vars(val_dataset.examples[-1]))
    print(f'Number of training examples: {len(train_dataset)}')
    print(f'Number of validation examples: {len(val_dataset)}')
    print(f'Number of testing examples: {len(test_dataset)}')

    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(train_dataset, max_size = MAX_VOCAB_SIZE)
    LABEL.build_vocab(train_dataset)

    text_vocab_size = len(TEXT.vocab)
    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

    train_data_iter = BucketIterator(train_dataset, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                      device=device, train=True)
    valid_data_iter = BucketIterator(val_dataset, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                     device=device, train=True)
    test_data_iter = BucketIterator(test_dataset, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                    device=device, train=False)

    # train_data_iter, valid_data_iter, test_data_iter = BucketIterator.splits(
    #     (train_dataset, val_dataset, test_dataset),
    #     batch_size=batch_size,
    #     device=device)
    print(TEXT.vocab.freqs.most_common(20))
    print(TEXT.vocab.itos[:10])
    print(LABEL.vocab.stoi)

    return train_data_iter, valid_data_iter, test_data_iter, text_vocab_size

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


#create model
train_data_iter, valid_data_iter, test_data_iter, text_vocab_size = data_load()
INPUT_DIM = text_vocab_size
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)



best_valid_loss = float('inf')

for epoch in range(num_epoch):

    start_time = time.time()

    train_loss, train_acc = train(model, train_data_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_data_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), path + '/nlp/model/nlp1-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')