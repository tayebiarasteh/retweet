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
LABEL = data.LabelField()

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

    fields = [('id', None), ('user_id', None), ('text', TEXT), ('label', LABEL)]

    train_dataset = TabularDataset(path=path + '/nlp/data/train_SemEval.csv',
                                            format='csv', skip_header=True, fields=fields)
    test_dataset = TabularDataset(path=path + '/nlp/data/test_SemEval.csv',
                                  format='csv', skip_header=True, fields=fields)

    print(vars(train_dataset.examples[0]))
    print(vars(test_dataset.examples[-1]))
    print(f'Number of training examples: {len(train_dataset)}')
    print(f'Number of testing examples: {len(test_dataset)}')

    MAX_VOCAB_SIZE = 25_000

    TEXT.build_vocab(train_dataset,
                     max_size=MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_dataset)

    text_vocab_size = len(TEXT.vocab)
    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")


    train_data_iter = BucketIterator(train_dataset, batch_size=batch_size, train=True, sort_within_batch=True,
                                     sort_key=lambda x: len(x.text), device=device
                                     )
    test_data_iter = BucketIterator(test_dataset, batch_size=batch_size, sort_within_batch=True,
                                     sort_key=lambda x: len(x.text), device=device, train=False)

    return train_data_iter, test_data_iter, text_vocab_size


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        predictions = model(batch.text)

        loss = criterion(predictions, batch.label)

        acc = categorical_accuracy(predictions, batch.label)

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
            predictions = model(batch.text)

            loss = criterion(predictions, batch.label)

            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


#create model
train_data_iter, test_data_iter, text_vocab_size = data_load()
INPUT_DIM = text_vocab_size
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = len(LABEL.vocab)

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)



best_valid_loss = float('inf')

for epoch in range(num_epoch):

    start_time = time.time()

    train_loss, train_acc = train(model, train_data_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_data_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), path + '/nlp/model/nlp_RNN.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')