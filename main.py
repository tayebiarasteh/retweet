'''
This is the main function running the Training, Validation, Testing process.
Set the hyper-parameters and model parameters here. [data parameters from config file]

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
'''

# Deep Learning Modules
from torch.nn import *
import torch
import torch.optim as optim
import spacy

# User Defined Modules
from configs.serde import *
from Train_Test_Valid import Training, Prediction, Mode
from data.data_handler import *
from models.biLSTM import *

#System Modules
from itertools import product
import time
import csv
import warnings
warnings.filterwarnings('ignore')



def main_train():
    '''Main function for training + validation.'''

    # if we are resuming training on a model
    RESUME = True

    # Hyper-parameters
    NUM_EPOCH = 100
    LOSS_FUNCTION = CrossEntropyLoss
    OPTIMIZER = optim.Adam
    BATCH_SIZE = 32
    #max_vocab_size: takes the 25000 most frequent words as the vocab
    MAX_VOCAB_SIZE = 25000
    lr = 5e-5
    EXPERIMENT_NAME = "Adam_lr" + str(lr) + "_max_vocab_size" + str(MAX_VOCAB_SIZE)

    if RESUME == True:
        params = open_experiment(EXPERIMENT_NAME)
    else:
        # put the new experiment name here.
        params = create_experiment(EXPERIMENT_NAME)
    cfg_path = params["cfg_path"]

    # Prepare data
    data_handler = data_provider_V2(cfg_path=cfg_path, batch_size=BATCH_SIZE,
                                    split_ratio=0.8, max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TRAIN)
    train_iterator, valid_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings = data_handler.data_loader()
    # Initialize trainer
    trainer = Training(cfg_path, num_epochs=NUM_EPOCH, RESUME=RESUME)

    # Model parameters
    optimiser_params = {'lr': lr, 'weight_decay': 1e-5}
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3
    MODEL = biLSTM(vocab_size=vocab_size, embeddings=pretrained_embeddings, embedding_dim=EMBEDDING_DIM,
                   hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
    if RESUME == True:
        trainer.load_checkpoint(model=MODEL, optimiser=OPTIMIZER,
                        optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION)
    else:
        trainer.setup_model(model=MODEL, optimiser=OPTIMIZER,
                        optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION)
    # Execute Training
    trainer.execute_training(train_loader=train_iterator, valid_loader=valid_iterator, batch_size=BATCH_SIZE)



def main_test():
    '''Main function for testing'''
    # Configs
    EXPERIMENT_NAME = 'Adam_lr5e-05_max_vocab_size25000'
    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']

    # Hyper-parameters
    BATCH_SIZE = 32

    # Prepare data
    # use the same "max_vocab_size" as in training
    data_handler_test = data_provider_V2(cfg_path=cfg_path, batch_size=BATCH_SIZE,
                                         max_vocab_size=25000, mode=Mode.TEST)
    test_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings = data_handler_test.data_loader()
    # Initialize predictor
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=biLSTM, vocab_size=vocab_size,
                          embeddings=pretrained_embeddings, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
    # Execute Testing
    predictor.predict(test_iterator, batch_size=BATCH_SIZE)



def main_manual_predict(PHRASE=None, prediction_mode='Manualpart1'):
    '''
    Manually predicts the polarity of the given sentence.
    :mode:
        'Manualpart1' predicts the sentiment of the tweet (part 1 pf the project)
        'Manualpart2' predicts the semtiment of the potential reply of the tweet (part 2)
        Note that for each part you should give the correct experiment name to load the correct model for it.
    '''
    if PHRASE == None:
        # Enter your phrase below here:
        PHRASE = "it is just fuckin awesome!"

    # Configs
    start_time = time.time()
    if prediction_mode == 'Manualpart1':
        EXPERIMENT_NAME = 'Adam_lr5e-05_max_vocab_size25000'
    elif prediction_mode == 'Manualpart2':
        EXPERIMENT_NAME = 'POSTREPLY_Adam_lr0.0002_max_vocab_size25000'

    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']

    # Prepare the network parameters
    # use the same "max_vocab_size" as in training
    if prediction_mode == 'Manualpart1':
        data_handler_test = data_provider_V2(cfg_path=cfg_path,
                                             max_vocab_size=25000, mode=Mode.PREDICTION)
    elif prediction_mode == 'Manualpart2':
        data_handler_test = data_provider_PostReply(cfg_path=cfg_path,
                                                    split_ratio=0.8, max_vocab_size=25000, mode=Mode.PREDICTION)

    labels, vocab_idx, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings = data_handler_test.data_loader()

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=biLSTM, vocab_size=vocab_size,
                          embeddings=pretrained_embeddings, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
    # Execute Prediction
    predictor.manual_predict(labels=labels, vocab_idx=vocab_idx,
                             phrase=PHRASE, mode=Mode.PREDICTION, prediction_mode=prediction_mode)
    # Duration
    end_time = time.time()
    test_mins, test_secs = prediction_time(start_time, end_time)
    print(f'Prediction Time: {test_mins}m {test_secs}s')



def main_tweet_reply_manual_predict(PHRASE=None):
    '''Manually predicts the reply sentiment that would be taught to receive.'''

    if PHRASE == None:
        # Enter your phrase below here:
        PHRASE = "How you doin? :D"

    # Configs
    start_time = time.time()
    EXPERIMENT_NAME = 'Adam_lr5e-05_max_vocab_size25000'
    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']
    # Prepare the network parameters
    # use the same "max_vocab_size" as in training
    data_handler_test = data_provider_V2(cfg_path=cfg_path, max_vocab_size=25000, mode=Mode.PREDICTION)
    labels, vocab_idx, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings = data_handler_test.data_loader()
    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=biLSTM, vocab_size=vocab_size,
                          embeddings=pretrained_embeddings, pad_idx=PAD_IDX, unk_idx=UNK_IDX)

    # TODO: load the second model here
    # Execute Prediction
    predictor.manual_predict(labels=labels, vocab_idx=vocab_idx, phrase=PHRASE, mode=Mode.PREDICTION)
    # Duration
    end_time = time.time()
    test_mins, test_secs = prediction_time(start_time, end_time)
    print(f'Prediction Time: {test_mins}m {test_secs}s')




def main_reply_predict():
    '''
    Manually predicts the polarity of the given replies,
    which will be regarded as the labels for the corresponding tweets.
    Note: first you need to create a csv file which you want to save the labels in.
    '''
    # Configs
    start_time = time.time()
    EXPERIMENT_NAME = 'Adam_lr5e-05_max_vocab_size25000'
    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']

    # Prepare the network parameters (use the same "max_vocab_size" as in training)
    data_handler_test = data_provider_V2(cfg_path=cfg_path, max_vocab_size=25000, mode=Mode.PREDICTION)
    labels, vocab_idx, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings = data_handler_test.data_loader()

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=biLSTM, vocab_size=vocab_size,
                          embeddings=pretrained_embeddings, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
    reply_dataset = []
    with open(os.path.join(params['postreply_data_path'], params['reply_file_name'])) as csv_file:
        data = csv.reader(csv_file)
        for row in data:
            reply_dataset.append(row)
    reply_dataset[0][0] = 'label'

    # Execute Prediction
    nlp = spacy.load('en')
    for idx, item in enumerate(reply_dataset):
        if idx != 0:
            PHRASE = item[4]
            reply_dataset[idx][0] = predictor.manual_predict(labels=labels, vocab_idx=vocab_idx, phrase=PHRASE,
                                                             tokenizer=nlp, mode=Mode.REPLYPREDICTION)
    # Writing the labels to a csv file
    with open(os.path.join(params['postreply_data_path'], params['reply_with_label_file_name']), 'w') as myfile:
        updated_reply_dataset = csv.writer(myfile)
        for row in reply_dataset:
            updated_reply_dataset.writerow(row)

    # Removing the repetitions
    summarizer(data_path=params['postreply_data_path'],
               input_file_name=params['reply_with_label_file_name'],
               output_file_name=params['reply_with_max_label_file_name'])
    # Duration
    end_time = time.time()
    test_mins, test_secs = prediction_time(start_time, end_time)
    print(f'Total Time: {test_mins}m {test_secs}s')



def main_train_postreply():
    '''
    Main function for training + validation of the second part of the project:
    Sentiment analysis of the Post-Replies.
    '''
    # if we are resuming training on a model
    RESUME = False

    # Hyper-parameters
    NUM_EPOCH = 100
    LOSS_FUNCTION = CrossEntropyLoss
    OPTIMIZER = optim.Adam
    BATCH_SIZE = 32
    #max_vocab_size: takes the 25000 most frequent words as the vocab
    MAX_VOCAB_SIZE = 25000
    lr = 2e-4
    EXPERIMENT_NAME = "POSTREPLY_Adam_lr" + str(lr) + "_max_vocab_size" + str(MAX_VOCAB_SIZE)

    if RESUME == True:
        params = open_experiment(EXPERIMENT_NAME)
    else:
        # put the new experiment name here.
        params = create_experiment(EXPERIMENT_NAME)
    cfg_path = params["cfg_path"]

    # Prepare data
    data_handler = data_provider_PostReply(cfg_path=cfg_path, batch_size=BATCH_SIZE,
                                    split_ratio=0.8, max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TRAIN)
    train_iterator, valid_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings = data_handler.data_loader()
    # Initialize trainer
    trainer = Training(cfg_path, num_epochs=NUM_EPOCH, RESUME=RESUME)

    # Model parameters
    optimiser_params = {'lr': lr}
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3
    MODEL = biLSTM(vocab_size=vocab_size, embeddings=pretrained_embeddings, embedding_dim=EMBEDDING_DIM,
                   hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
    if RESUME == True:
        trainer.load_checkpoint(model=MODEL, optimiser=OPTIMIZER,
                        optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION)
    else:
        trainer.setup_model(model=MODEL, optimiser=OPTIMIZER,
                        optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION)
    # Execute Training
    trainer.execute_training(train_loader=train_iterator, valid_loader=valid_iterator, batch_size=BATCH_SIZE)



def prediction_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def experiment_deleter():
    '''To delete an experiment and reuse the same experiment name'''
    parameters = dict(lr = [5e-4], max_vocab_size = [25000])
    param_values = [v for v in parameters.values()]
    for lr, MAX_VOCAB_SIZE in product(*param_values):
        delete_experiment("new2Adam_lr" + str(lr) + "_max_vocab_size" + str(MAX_VOCAB_SIZE))



if __name__ == '__main__':
    # experiment_deleter()
    # main_train()
    # main_test()
    # main_manual_predict(prediction_mode='Manualpart1')
    # main_reply_predict()
    main_train_postreply()