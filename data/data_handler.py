"""
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""

import numpy as np
import random
import os
import csv
import torch
from torch.utils.data import Dataset
from Training import Mode
from configs.serde import read_config


class data_provider(Dataset):
    def __init__(self, cfg_path, dataset_name, size, mode=Mode.TRAIN, seed=1):
        '''
        Args:
            cfg_path (string):
                Config file path of the experiment
            size (int):
                No of messages to be used from the dataset
            mode (enumeration Mode):
                Nature of operation to be done with the data.
                Possible inputs are Mode.PREDICT, Mode.TRAIN, Mode.VALID
                Default value: Mode.TRAIN
        '''
        params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.mode = mode
        self.datset_name = dataset_name
        self.dataset_path = os.path.join(params['input_data_path'], dataset_name)
        self.testset_path = os.path.join(params['test_data_path'], dataset_name)
        self.size = size
        self.input_tensor, self.target_tensor = self._init_dataset(seed)


    def __len__(self):
        '''Returns length of the dataset'''
        return self.size

    def __getitem__(self, idx):
        '''
        Using self.input_tensor list and self.target_tensor list the argument value idx,
        return inputs and labels(if applicable based on Mode)
        The inputs and labels are returned in torch tensor format.
        '''
        X = self.input_tensor[idx]
        Y = self.target_tensor[idx]

        # Converting input and label to tensor
        x_input = torch.from_numpy(X)
        y_input = torch.from_numpy(Y)

        if self.mode == Mode.PREDICT:
            return x_input
        else:
            return x_input, y_input


    def _init_dataset(self, seed):
        ''' Collects the input and label arrays and saves them to a list.
        @inspired_from: Elvis Saravia
        '''
        # Dataset path
        if self.mode==Mode.VALID:
            dataset_path = self.testset_path
        else:
            dataset_path = self.dataset_path

        # Opening the data file
        # SIDs = [] # tweet id
        # UIDs = [] # user id
        labels = []
        messages = []
        with open(dataset_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for SID, UID, label, message in reader:
                # SIDs.append(SID)
                # UIDs.append(UID)
                labels.append(label)
                messages.append(message)

        if len(messages) < self.size:
            print('The size you requested is more than the total available messages.')
            self.size = len(messages)

        elif len(messages) > self.size:
            print('The size you requested is less than the total available messages. '
                  'The desired number of messages randomly will be picked.')
            random.seed(seed)
            random.shuffle(messages)
            messages = messages[:self.size]

        # Building Vocabulary
        vocab = set()
        word2idx = {}
        idx2word = {}

        # Updating with individual tokens
        for s in messages:
            vocab.update(s.split(' '))

        # Sort the vocab
        vocab = sorted(vocab)

        # Word to index mapping
        word2idx['<pad>'] = 0   # add a padding token with index 0
        for index, word in enumerate(vocab):
            word2idx[word] = index + 1  # +1 because of pad token

        # Index to word mapping
        for word, index in word2idx.items():
            idx2word[index] = word

        # Converting data into tensors
        input_tensor = [[word2idx[s] for s in es.split(' ')] for es in messages]

        # Padding data to be of equal sizes
        max_len = max(len(t) for t in input_tensor)
        def pad_sequences(x, max_len):
            padded = np.zeros((max_len), dtype=np.int64)
            if len(x) > max_len: padded[:] = x[:max_len]
            else: padded[:len(x)] = x
            return padded
        input_tensor = [pad_sequences(x, max_len) for x in input_tensor]

        # Converting targets to one-hot encoding vectors
        classes = list(set(labels))
        # Binarizer
        emotion_dict = {'positive': [0, 0, 0], 'negative': [0, 1, 0], 'neutral': [0, 0, 1]}

        target_tensor = np.zeros((len(labels), len(classes)))
        for i, label in enumerate(labels):
            target_tensor[i] = (emotion_dict[label])


        return input_tensor, target_tensor
