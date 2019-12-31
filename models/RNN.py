"""
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""

'''
GRU
'''

import torch
import torch.nn as nn
import pdb


class GRUU(nn.Module):
    def __init__(self, vocab_size, batch_size, embedding_dim=256, num_hidden_units=1024, output_size=3):
        '''
        :param vocab_size: len(idx2word) or len(word2idx)

        :param embedding_dim: Each word represented with 256 numbers, so embedding_dim means
         how many numbers you want to represent your token with. Each time we run this function,
         we get a new embedding values, but it's fixed (each number has a fixed representation everywhere).

        :param output_size: number of output classes
        '''
        super(GRUU, self).__init__()
        self.batch_size = batch_size
        self.num_hidden_units = num_hidden_units
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.output_size = output_size

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.gru = nn.GRU(self.embedding_dim, self.num_hidden_units)
        self.fc = nn.Linear(self.num_hidden_units, self.output_size)

    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch_size, self.num_hidden_units)).to(device)

    def forward(self, x, device):
        x = self.embedding(x)
        self.hidden = torch.zeros((1, self.batch_size, self.num_hidden_units)).to(device)
        output, self.hidden = self.gru(x, self.hidden)  # max_len X batch_size X hidden_units
        out = output[-1, :, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out, self.hidden