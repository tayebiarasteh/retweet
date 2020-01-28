'''
This is the main function running the Training, Validation, Testing process.
Set the hyper-parameters and model parameters here. [data parameters from config file]
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
'''

# Deep Learning Modules
from torch.nn import *
import torch
import torch.optim as optim

# User Defined Modules
from configs.serde import *
from Train_Test_Valid import Training, Prediction
from data.data_handler import *
from models.biLSTM import *

#System Modules
from itertools import product
import warnings
warnings.filterwarnings('ignore')


def main_train():
    '''Main function for training + validation.'''

    '''Hyper-parameters'''
    NUM_EPOCH = 10
    LOSS_FUNCTION = CrossEntropyLoss
    OPTIMIZER = optim.Adam
    BATCH_SIZE = 32
    #max_vocab_size: takes the 25000 most frequent words as the vocab
    parameters = dict(lr = [5e-4], max_vocab_size = [25000])
    param_values = [v for v in parameters.values()]

    '''Hyper-parameter testing'''
    for lr, MAX_VOCAB_SIZE in product(*param_values):
        # put the new experiment name here.
        params = create_experiment("Adam_lr" + str(lr) + "_max_vocab_size" + str(MAX_VOCAB_SIZE))
        cfg_path = params["cfg_path"]

        '''Prepare data'''
        data_handler = data_provider_V2(cfg_path=cfg_path, batch_size=BATCH_SIZE,
                                        split_ratio=0.8, max_vocab_size=MAX_VOCAB_SIZE)
        train_iterator, valid_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings = data_handler.data_loader()

        '''Initialize trainer'''
        trainer = Training(cfg_path)

        '''Model parameters'''
        optimiser_params = {'lr': lr}
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        OUTPUT_DIM = 3
        MODEL = biLSTM(vocab_size=vocab_size, embeddings=pretrained_embeddings, embedding_dim=EMBEDDING_DIM,
                       hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
        trainer.setup_model(model=MODEL, optimiser=OPTIMIZER,
                            optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION)
        '''Execute Training'''
        trainer.execute_training(train_loader=train_iterator, valid_loader=valid_iterator, num_epochs=NUM_EPOCH)



def main_test():
    '''Main function for prediction'''
    # Configs
    EXPERIMENT_NAME = 'Adam_lr0.0005_max_vocab_size25000'
    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']

    '''Hyper-parameters'''
    BATCH_SIZE = 32

    '''Prepare data'''
    # use the same "max_vocab_size" as in training
    data_handler_test = data_provider_V2(cfg_path=cfg_path, batch_size=BATCH_SIZE,
                                         max_vocab_size=25000, mode=Mode.PREDICT)
    test_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings = data_handler_test.data_loader()

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=biLSTM, vocab_size=vocab_size,
                          embeddings=pretrained_embeddings, pad_idx=PAD_IDX, unk_idx=UNK_IDX)

    '''Execute Prediction'''
    predictor.predict(test_iterator)



def experiment_deleter():
    '''To delete an experiment and reuse the same experiment name'''
    parameters = dict(lr = [5e-4], max_vocab_size = [25000])
    param_values = [v for v in parameters.values()]
    for lr, MAX_VOCAB_SIZE in product(*param_values):
        delete_experiment("Adam_lr" + str(lr) + "_max_vocab_size" + str(MAX_VOCAB_SIZE))



if __name__ == '__main__':
    # experiment_deleter()
    # main_train()
    main_test()
