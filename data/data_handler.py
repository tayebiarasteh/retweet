"""
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""

import numpy as np
import random
import torch
from torchtext import data
from Train_Test import Mode
from configs.serde import read_config
import pdb




class data_provider_V2():
    '''
    Packed padded sequences
    Pre-trained embedding: glove.6B.100d
    tokenizer: spacy
    '''
    def __init__(self, cfg_path, batch_size, split_ratio=0.8, max_vocab_size=25000, mode=Mode.TRAIN, seed=1):
        '''
        Args:
            cfg_path (string):
                Config file path of the experiment
            max_vocab_size (int):
                The number of unique words in our training set is usually over 100,000,
                which means that our one-hot vectors will have over 100,000 dimensions! SLOW TRAINIG!
                We only take the top max_vocab_size most common words.
            split_ratio (float):
                train-valid splitting
            mode (enumeration Mode):
                Nature of operation to be done with the data.
                Possible inputs are Mode.PREDICT, Mode.TRAIN, Mode.VALID
                Default value: Mode.TRAIN
        '''
        params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.mode = mode
        self.seed = seed
        self.split_ratio = split_ratio
        self.max_vocab_size = max_vocab_size
        self.dataset_path = params['input_data_path']
        self.batch_size = batch_size


    def data_loader(self):
        '''
        :include_lengths: Packed padded sequences: will make our RNN only process the non-padded elements of our sequence,
            and for any padded element the `output` will be a zero tensor.
        :tokenize: the "tokenization" (the act of splitting the string into discrete "tokens") should be done using the spaCy tokenizer.
        '''
        '''Packed padded sequences'''
        TEXT = data.Field(tokenize='spacy', include_lengths=True)  # For saving the length of sentences
        LABEL = data.LabelField()

        fields = [('id', None), ('user_id', None),  ('label', LABEL), ('text', TEXT)]
        train_data, test_data = data.TabularDataset.splits(
            path=self.dataset_path,
            train='train_and_dev_b_2014.txt',
            test='test_gold_b_2014_2015.txt',
            format='tsv',  # tab separated value
            fields=fields,
            skip_header=False)

        # validation data
        train_data, valid_data = train_data.split(random_state=random.seed(self.seed), split_ratio=self.split_ratio)

        # create the vocabulary only on the training set!!!
        # vectors: instead of having our word embeddings initialized randomly, they are initialized with these pre-trained vectors.
        # initialize words in your vocabulary but not in your pre-trained embeddings to Gaussian
        TEXT.build_vocab(train_data, max_size=self.max_vocab_size,
                         vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
        LABEL.build_vocab(train_data)

        vocab_size = len(TEXT.vocab)

        # for packed padded sequences all of the tensors within a batch need to be sorted by their lengths
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((
            train_data, valid_data, test_data), batch_size=self.batch_size,
            sort_within_batch=True, sort_key=lambda x: len(x.text))

        return train_iterator, valid_iterator, test_iterator, vocab_size




if __name__=='__main__':
    CONFIG_PATH = '/home/soroosh/Documents/Repositories/twitter_sentiment/configs/config.json'
    data_handler = data_provider_V2(cfg_path=CONFIG_PATH, batch_size=1, split_ratio=0.8, max_vocab_size=25000)
    train_iterator, valid_iterator, test_iterator, vocab_size = data_handler.data_loader()
    # pdb.set_trace()
    # a=2