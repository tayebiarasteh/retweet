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
from models.CNN import *

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
    MAX_VOCAB_SIZE = 50000 #max_vocab_size: takes the 100,000 most frequent words as the vocab
    lr = 9e-5
    optimiser_params = {'lr': lr, 'weight_decay': 1e-5}
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256  # for the LSTM model:
    OUTPUT_DIM = 3
    MODEL_MODE = 'RNN' # 'RNN' or 'CNN'
    conv_out_ch = 200  # for the CNN model:
    filter_sizes = [3, 4, 5]  # for the CNN model:
    SPLIT_RATIO = 0.8 # ratio of the train set, 1.0 means 100% training, 0% valid data
    EXPERIMENT_NAME = "Adam_lr" + str(lr) + "_max_vocab_size" + str(MAX_VOCAB_SIZE)

    if RESUME == True:
        params = open_experiment(EXPERIMENT_NAME)
    else:
        params = create_experiment(EXPERIMENT_NAME)
    cfg_path = params["cfg_path"]

    # Prepare data
    data_handler = data_provider_V2(cfg_path=cfg_path, batch_size=BATCH_SIZE, split_ratio=SPLIT_RATIO,
                                    max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TRAIN, model_mode=MODEL_MODE)
    train_iterator, valid_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, weights = data_handler.data_loader()

    print(f'\nSummary:\n----------------------------------------------------')
    print(f'Total # of Training tweets: {BATCH_SIZE * len(train_iterator):,}')
    if SPLIT_RATIO == 1:
        print(f'Total # of Valid. tweets:   {0}')
    else:
        print(f'Total # of Valid. tweets:   {BATCH_SIZE * len(valid_iterator):,}')

    # Initialize trainer
    trainer = Training(cfg_path, num_epochs=NUM_EPOCH, RESUME=RESUME, model_mode=MODEL_MODE)
    if MODEL_MODE == 'RNN':
        MODEL = biLSTM(vocab_size=vocab_size, embeddings=pretrained_embeddings, embedding_dim=EMBEDDING_DIM,
                       hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
    elif MODEL_MODE == 'CNN':
        MODEL = CNN1d(vocab_size=vocab_size, embeddings=pretrained_embeddings, embedding_dim=EMBEDDING_DIM,
                       conv_out_ch=conv_out_ch, filter_sizes=filter_sizes, output_dim=OUTPUT_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)

    if RESUME == True:
        trainer.load_checkpoint(model=MODEL, optimiser=OPTIMIZER,
                        optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION, weight=weights)
    else:
        trainer.setup_model(model=MODEL, optimiser=OPTIMIZER,
                        optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION, weight=weights)
    trainer.execute_training(train_loader=train_iterator, valid_loader=valid_iterator, batch_size=BATCH_SIZE)



def main_test():
    '''Main function for testing'''
    EXPERIMENT_NAME = 'Adam_lr9e-05_max_vocab_size50000'
    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']

    # Hyper-parameters
    BATCH_SIZE = 32
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256
    MAX_VOCAB_SIZE = 50000  # use the same "max_vocab_size" as in training
    SPLIT_RATIO = 0.8 # use the same as in training.
    MODEL_MODE = 'RNN' # 'RNN' or 'CNN'

    # Prepare data
    data_handler_test = data_provider_V2(cfg_path=cfg_path, batch_size=BATCH_SIZE, split_ratio=SPLIT_RATIO,
                                         max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TEST, model_mode=MODEL_MODE)
    test_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings = data_handler_test.data_loader()
    # Initialize predictor
    predictor = Prediction(cfg_path, model_mode=MODEL_MODE)
    predictor.setup_model(model=biLSTM, vocab_size=vocab_size, embeddings=pretrained_embeddings,
                          embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
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
        PHRASE = "I have to say that I got divorced"

    # Configs
    start_time = time.time()
    if prediction_mode == 'Manualpart1':
        EXPERIMENT_NAME = 'Adam_lr5e-05_max_vocab_size25000'
    elif prediction_mode == 'Manualpart2':
        EXPERIMENT_NAME = 'POSTREPLY_Adam_lr9e-05_max_vocab_size100000'

    # Hyper-parameters
    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']
    MODEL_MODE = 'RNN' # 'RNN' or 'CNN'
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256
    MAX_VOCAB_SIZE = 50000  # use the same "max_vocab_size" as in training
    SPLIT_RATIO = 0.8 # use the same as in training.
    # Prepare the network parameters
    if prediction_mode == 'Manualpart1':
        data_handler_test = data_provider_V2(cfg_path=cfg_path, split_ratio=SPLIT_RATIO,
                                                    max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.PREDICTION, model_mode=MODEL_MODE)
    elif prediction_mode == 'Manualpart2':
        data_handler_test = data_provider_PostReply(cfg_path=cfg_path, split_ratio=SPLIT_RATIO,
                                                    max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.PREDICTION, model_mode=MODEL_MODE)

    labels, vocab_idx, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings = data_handler_test.data_loader()

    # Initialize prediction
    predictor = Prediction(cfg_path, model_mode=MODEL_MODE)
    predictor.setup_model(model=biLSTM, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                          embeddings=pretrained_embeddings, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
    # Execute Prediction
    predictor.manual_predict(labels=labels, vocab_idx=vocab_idx,
                             phrase=PHRASE, mode=Mode.PREDICTION, prediction_mode=prediction_mode)
    # Duration
    end_time = time.time()
    test_mins, test_secs = prediction_time(start_time, end_time)
    print(f'Prediction Time: {test_mins}m {test_secs}s')



def main_reply_predict():
    '''
    Manually predicts the polarity of the given replies,
    which will be regarded as the labels for the corresponding tweets
    and creates a labeled dataset of only tweets and corresponding labels.
    Note: first you need to create a csv file which you want to save
    the labels in ("data_post_reply_withlabel.csv").
    '''
    start_time = time.time()
    EXPERIMENT_NAME = 'Adam_lr9e-05_max_vocab_size50000'
    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']

    # Hyper-parameters
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256
    MAX_VOCAB_SIZE = 50000  # use the same "max_vocab_size" as in training
    MODEL_MODE = 'RNN' # 'RNN' or 'CNN'

    # Prepare the network parameters
    data_handler_test = data_provider_V2(cfg_path=cfg_path, max_vocab_size=MAX_VOCAB_SIZE,
                                         mode=Mode.PREDICTION, model_mode=MODEL_MODE)
    labels, vocab_idx, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings = data_handler_test.data_loader()

    # Initialize prediction
    predictor = Prediction(cfg_path, model_mode=MODEL_MODE)
    predictor.setup_model(model=biLSTM, vocab_size=vocab_size, embeddings=pretrained_embeddings,
                          embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
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
               output_file_name=params['final_data_post_reply_file_name'])
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
    MAX_VOCAB_SIZE = 100000 #max_vocab_size: takes the 100,000 most frequent words as the vocab
    lr = 9e-5
    optimiser_params = {'lr': lr, 'weight_decay': 1e-5}
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256
    OUTPUT_DIM = 3
    MODEL_MODE = 'RNN' # 'RNN' or 'CNN'
    conv_out_ch = 200  # for the CNN model:
    filter_sizes = [3, 4, 5]  # for the CNN model:
    SPLIT_RATIO = 0.8 # ratio of the train set, 1.0 means 100% training, 0% valid data
    EXPERIMENT_NAME = "noWeight_POSTREPLY_Adam_lr" + str(lr) + "_max_vocab_size" + str(MAX_VOCAB_SIZE)

    if RESUME == True:
        params = open_experiment(EXPERIMENT_NAME)
    else:
        params = create_experiment(EXPERIMENT_NAME)
    cfg_path = params["cfg_path"]

    # Prepare data
    data_handler = data_provider_PostReply(cfg_path=cfg_path, batch_size=BATCH_SIZE, split_ratio=SPLIT_RATIO,
                                           max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TRAIN, model_mode=MODEL_MODE)
    train_iterator, valid_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, weights = data_handler.data_loader()

    print(f'\nSummary:\n----------------------------------------------------')
    print(f'Total # of Training tweets: {BATCH_SIZE * len(train_iterator):,}')
    if SPLIT_RATIO == 1:
        print(f'Total # of Valid. tweets:   {0}')
    else:
        print(f'Total # of Valid. tweets:   {BATCH_SIZE * len(valid_iterator):,}')

    # Initialize trainer
    trainer = Training(cfg_path, num_epochs=NUM_EPOCH, RESUME=RESUME, model_mode=MODEL_MODE)

    if MODEL_MODE == 'RNN':
        MODEL = biLSTM(vocab_size=vocab_size, embeddings=pretrained_embeddings, embedding_dim=EMBEDDING_DIM,
                       hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
    elif MODEL_MODE == 'CNN':
        MODEL = CNN1d(vocab_size=vocab_size, embeddings=pretrained_embeddings, embedding_dim=EMBEDDING_DIM,
                       conv_out_ch=conv_out_ch, filter_sizes=filter_sizes, output_dim=OUTPUT_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)

    if RESUME == True:
        trainer.load_checkpoint(model=MODEL, optimiser=OPTIMIZER,
                        optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION, weight=weights)
    else:
        trainer.setup_model(model=MODEL, optimiser=OPTIMIZER,
                        optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION, weight=weights)
    trainer.execute_training(train_loader=train_iterator, valid_loader=valid_iterator, batch_size=BATCH_SIZE)



def main_test_postreply():
    '''Main function for testing of the second part of the project
    Sentiment analysis of the Post-Replies.
    '''
    EXPERIMENT_NAME = 'POSTREPLY_Adam_lr9e-05_max_vocab_size100000'
    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']

    # Hyper-parameters
    BATCH_SIZE = 32
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256
    MAX_VOCAB_SIZE = 100000  # use the same "max_vocab_size" as in training
    SPLIT_RATIO = 0.8 # use the same as in training.
    MODEL_MODE = 'RNN' # 'RNN' or 'CNN'

    # Prepare data
    data_handler_test = data_provider_PostReply(cfg_path=cfg_path, batch_size=BATCH_SIZE, split_ratio=SPLIT_RATIO,
                                         max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TEST, model_mode=MODEL_MODE)
    test_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings = data_handler_test.data_loader()
    # Initialize predictor
    predictor = Prediction(cfg_path, model_mode=MODEL_MODE)
    predictor.setup_model(model=biLSTM, vocab_size=vocab_size, embeddings=pretrained_embeddings,
                          embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
    predictor.predict(test_iterator, batch_size=BATCH_SIZE)




def prediction_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def experiment_deleter():
    '''To delete an experiment and reuse the same experiment name'''
    parameters = dict(lr = [9e-5], max_vocab_size = [100000])
    param_values = [v for v in parameters.values()]
    for lr, MAX_VOCAB_SIZE in product(*param_values):
        delete_experiment("CNN_Adam_lr" + str(lr) + "_max_vocab_size" + str(MAX_VOCAB_SIZE))



if __name__ == '__main__':
    # experiment_deleter()
    # main_train()
    # main_test()
    # main_manual_predict(prediction_mode='Manualpart2')
    # main_reply_predict()
    # main_train_postreply()
    main_test_postreply()