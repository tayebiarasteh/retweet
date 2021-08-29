"""
CNN model for the text data.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
https://tayebiarasteh.com/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class CNN1d(nn.Module):
    def __init__(self, vocab_size, embeddings, embedding_dim=200,
                 conv_out_ch=200, filter_sizes=[3,4,5], output_dim=3, pad_idx=1, unk_idx=0):
        '''
        :pad_idx: the index of the padding token <pad> in the vocabulary
        :conv_out_ch: number of the different kernels.
        :filter_sizes: a list of different kernel sizes we use here.
        '''
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # replace the initial weights of the `embedding` layer with the pre-trained embeddings.
        self.embedding.weight.data.copy_(embeddings)
        # these are irrelevant for determining sentiment:
        self.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
        self.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=conv_out_ch,
                      kernel_size=fs) for fs in filter_sizes])

        self.fc = nn.Linear(len(filter_sizes) * conv_out_ch, output_dim)
        self.dropout = nn.Dropout(0.5)


    def forward(self, text):
        # text = [batch size, sent len]

        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)
        # embedded = [batch size, emb dim, sent len]

        # pad if the length of the sentence is less than the kernel size
        if embedded.shape[2] < 5:
            dif = 5 - embedded.shape[2]
            embedded = F.pad(embedded, (0, dif), "constant", 0)

        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)
