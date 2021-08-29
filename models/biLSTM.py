"""
dropout + Bidirectional LSTM with 2 layers + dropout + fully connected.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
https://tayebiarasteh.com/
"""

import torch
import torch.nn as nn
import pdb


class biLSTM(nn.Module):
    def __init__(self, vocab_size, embeddings=None, embedding_dim=100, hidden_dim=256, output_dim=3, pad_idx=1, unk_idx=0):
        '''
        :pad_idx: the index of the padding token <pad> in the vocabulary
        :num_layers: number of biLSTMs stacked on top of each other
        '''
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # replace the initial weights of the `embedding` layer with the pre-trained embeddings.
        self.embedding.weight.data.copy_(embeddings)
        # these are irrelevant for determining sentiment:
        self.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
        self.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                           bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        # Note: never use dropout on the input or output layers (text or fc in this case),
        # you only ever want to use dropout on intermediate layers.

        self.fc = nn.Linear(hidden_dim * 2, output_dim)


    def forward(self, text, text_lengths):
        '''
        In some frameworks you must feed the initial hidden state, $h_0$, into the RNN,
        however in PyTorch, if no initial hidden state is passed as an argument it defaults to a tensor of all zeros.
        :nn.utils.rnn.pack_padded_sequence: This will cause our RNN to only process the non-padded elements of our sequence.
        '''
        # text : [sent len, batch size]

        embedded = self.dropout(self.embedding(text))
        # embedded : [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # output (packed_output) is the concatenation of the hidden state from every time step
        # hidden is simply the final hidden state.
        # hidden : [num layers * num directions, batch size, hid dim]
        # cell : [num layers * num directions, batch size, hid dim]

        # unpack sequence [not needed, only for demonstration]
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output : [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden : [batch size, hid dim * num directions]

        return self.fc(hidden)
