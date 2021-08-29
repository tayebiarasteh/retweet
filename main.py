'''
This is the main function running the Training, Validation, Testing process.
Set the hyper-parameters and model parameters here. [data parameters from config file]

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
https://tayebiarasteh.com/
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
from data.data_processing import *
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
    BATCH_SIZE = 256
    MAX_VOCAB_SIZE = 50000 #max_vocab_size: takes the 100,000 most frequent words as the vocab
    lr = 1e-4
    optimiser_params = {'lr': lr, 'weight_decay': 1e-5}
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256  # for the LSTM model:
    OUTPUT_DIM = 3
    MODEL_MODE = 'RNN' # 'RNN' or 'CNN'
    conv_out_ch = 200  # for the CNN model:
    filter_sizes = [3, 4, 5]  # for the CNN model:
    SPLIT_RATIO = 0.85 # ratio of the train set, 1.0 means 100% training, 0% valid data
    EXPERIMENT_NAME = "Adam_lr" + str(lr) + "_max_vocab_size" + str(MAX_VOCAB_SIZE)

    if RESUME == True:
        params = open_experiment(EXPERIMENT_NAME)
    else:
        params = create_experiment(EXPERIMENT_NAME)
    cfg_path = params["cfg_path"]

    # Prepare data
    data_handler = data_provider_V2(cfg_path=cfg_path, batch_size=BATCH_SIZE, split_ratio=SPLIT_RATIO,
                                    max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TRAIN, model_mode=MODEL_MODE)
    train_iterator, valid_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, weights, classes = data_handler.data_loader()

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
    EXPERIMENT_NAME = 'Adam_lr0.0001_max_vocab_size50000'
    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']

    # Hyper-parameters
    BATCH_SIZE = 256
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256
    MAX_VOCAB_SIZE = 50000  # use the same "max_vocab_size" as in training
    SPLIT_RATIO = 0.85 # use the same as in training.
    MODEL_MODE = 'RNN' # 'RNN' or 'CNN'

    # Prepare data
    data_handler_test = data_provider_V2(cfg_path=cfg_path, batch_size=BATCH_SIZE, split_ratio=SPLIT_RATIO,
                                         max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TEST, model_mode=MODEL_MODE)
    test_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, classes = data_handler_test.data_loader()
    # Initialize predictor
    predictor = Prediction(cfg_path, classes=classes, model_mode=MODEL_MODE)
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

    labels, vocab_idx, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, classes = data_handler_test.data_loader()

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



def main_reply_predict(DATA_MODE = 'getoldtweet'):
    '''
    Manually predicts the polarity of the given replies,
    which will be regarded as the labels for the corresponding tweets
    and creates a labeled dataset of only tweets and corresponding labels.
    :DATA_MODE: 'getoldtweet' or 'philipp'
    '''
    start_time = time.time()
    EXPERIMENT_NAME = 'Adam_lr0.0001_max_vocab_size50000'
    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']
    # Hyper-parameters
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 256
    MAX_VOCAB_SIZE = 50000  # use the same "max_vocab_size" as in training
    MODEL_MODE = 'RNN' # 'RNN' or 'CNN'

    if DATA_MODE == 'getoldtweet':
        original_data = params['reply_file_name']
        predicted_data = params['reply_with_label_file_name']
        final_data = params['final_data_post_reply_file_name']
    if DATA_MODE == 'philipp':
        original_data = params['philipp_data']
        predicted_data = params['philipp_with_label_file_name']
        final_data = params['philipp_final_post_reply_file_name']

    # Prepare the network parameters
    data_handler_test = data_provider_V2(cfg_path=cfg_path, max_vocab_size=MAX_VOCAB_SIZE,
                                         mode=Mode.PREDICTION, model_mode=MODEL_MODE)
    labels, vocab_idx, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, classes = data_handler_test.data_loader()

    # Initialize prediction
    predictor = Prediction(cfg_path, model_mode=MODEL_MODE, classes=classes)
    predictor.setup_model(model=biLSTM, vocab_size=vocab_size, embeddings=pretrained_embeddings,
                          embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)

    data = pd.read_csv(os.path.join(params['postreply_data_path'], original_data))
    data = data.reindex(columns=['label', 'tweet', 'id', 'user', 'reply'])
    # Execute Prediction
    nlp = spacy.load('en')
    for idx, item in enumerate(data['reply']):
        print(idx)
        PHRASE = str(item)
        data['label'][idx] = predictor.manual_predict(labels=labels, vocab_idx=vocab_idx, phrase=PHRASE,
                                                             tokenizer=nlp, mode=Mode.REPLYPREDICTION)
    data.to_csv(os.path.join(params['postreply_data_path'], predicted_data), index=False)

    # Removing the repetitions
    summarizer(data_path=params['postreply_data_path'],
               input_file_name=predicted_data,
               output_file_name=final_data)
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
    NUM_EPOCH = 500
    LOSS_FUNCTION = CrossEntropyLoss
    OPTIMIZER = optim.Adam
    BATCH_SIZE = 256
    MAX_VOCAB_SIZE = 750000 #max_vocab_size: takes the 100,000 most frequent words as the vocab
    lr = 9e-5
    optimiser_params = {'lr': lr, 'weight_decay': 1e-4}
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 300
    OUTPUT_DIM = 3
    MODEL_MODE = "CNN" # "RNN" or "CNN"
    conv_out_ch = 200  # for the CNN model:
    filter_sizes = [3, 4, 5]  # for the CNN model:
    SPLIT_RATIO = 0.9 # ratio of the train set, 1.0 means 100% training, 0% valid data
    EXPERIMENT_NAME = "new_october_CNN"

    if RESUME == True:
        params = open_experiment(EXPERIMENT_NAME)
    else:
        params = create_experiment(EXPERIMENT_NAME)
    cfg_path = params["cfg_path"]

    # Prepare data
    data_handler = data_provider_PostReply(cfg_path=cfg_path, batch_size=BATCH_SIZE, split_ratio=SPLIT_RATIO,
                                           max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TRAIN, model_mode=MODEL_MODE)
    train_iterator, valid_iterator, vocab_size, PAD_IDX, UNK_IDX, pretrained_embeddings, weights, classes = data_handler.data_loader()

    if SPLIT_RATIO == 1:
        total_valid_tweets = 0
    else:
        total_valid_tweets = BATCH_SIZE * len(valid_iterator)
    total_train_tweets = BATCH_SIZE * len(train_iterator)
    print(f'\nSummary:\n----------------------------------------------------')
    print(f'Total # of Training tweets: {total_train_tweets:,}')
    print(f'Total # of Valid. tweets:   {total_valid_tweets:,}')

    # Initialize trainer
    trainer = Training(cfg_path, num_epochs=NUM_EPOCH, RESUME=RESUME, model_mode=MODEL_MODE)

    if MODEL_MODE == "RNN":
        MODEL = biLSTM(vocab_size=vocab_size, embeddings=pretrained_embeddings, embedding_dim=EMBEDDING_DIM,
                       hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)
    elif MODEL_MODE == "CNN":
        MODEL = CNN1d(vocab_size=vocab_size, embeddings=pretrained_embeddings, embedding_dim=EMBEDDING_DIM,
                       conv_out_ch=conv_out_ch, filter_sizes=filter_sizes, output_dim=OUTPUT_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX)

    if RESUME == True:
        trainer.load_checkpoint(model=MODEL, optimiser=OPTIMIZER,
                        optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION, weight=weights)
    else:
        trainer.setup_model(model=MODEL, optimiser=OPTIMIZER,
                        optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION, weight=weights)
        # writes the params to config file
        params = read_config(cfg_path)
        params['Network']['vocab_size'] = vocab_size
        params['Network']['PAD_IDX'] = PAD_IDX
        params['Network']['UNK_IDX'] = UNK_IDX
        params['Network']['classes'] = classes
        params['Network']['SPLIT_RATIO'] = SPLIT_RATIO
        params['Network']['MAX_VOCAB_SIZE'] = MAX_VOCAB_SIZE
        params['Network']['HIDDEN_DIM'] = HIDDEN_DIM
        params['Network']['EMBEDDING_DIM'] = EMBEDDING_DIM
        params['Network']['conv_out_ch'] = conv_out_ch
        params['Network']['MODEL_MODE'] = MODEL_MODE
        params['total_train_tweets'] = total_train_tweets
        params['total_valid_tweets'] = total_valid_tweets
        write_config(params, cfg_path, sort_keys=True)

    trainer.execute_training(train_loader=train_iterator, valid_loader=valid_iterator, batch_size=BATCH_SIZE)



def main_test_postreply():
    '''Main function for testing of the second part of the project
    Sentiment analysis of the Post-Replies.
    '''
    EXPERIMENT_NAME = 'new_october_CNN'
    BATCH_SIZE = 256

    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']
    vocab_size = params['Network']['vocab_size']
    PAD_IDX = params['Network']['PAD_IDX']
    UNK_IDX = params['Network']['UNK_IDX']
    classes = params['Network']['classes']
    MAX_VOCAB_SIZE = params['Network']['MAX_VOCAB_SIZE']
    SPLIT_RATIO = params['Network']['SPLIT_RATIO']
    EMBEDDING_DIM = params['Network']['EMBEDDING_DIM']
    HIDDEN_DIM = params['Network']['HIDDEN_DIM']
    conv_out_ch = params['Network']['conv_out_ch']
    MODEL_MODE = params['Network']['MODEL_MODE']
    pretrained_embeddings = torch.zeros((vocab_size, EMBEDDING_DIM))

    # Prepare data
    data_handler_test = data_provider_PostReply(cfg_path=cfg_path, batch_size=BATCH_SIZE, split_ratio=SPLIT_RATIO,
                                         max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TEST, model_mode=MODEL_MODE)
    test_iterator = data_handler_test.data_loader()
    # Initialize predictor
    predictor = Prediction(cfg_path, model_mode=MODEL_MODE, classes=classes)

    if MODEL_MODE == "RNN":
        MODEL = biLSTM
    elif MODEL_MODE == "CNN":
        MODEL = CNN1d

    predictor.setup_model(model=MODEL, vocab_size=vocab_size, embeddings=pretrained_embeddings,
                          embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX,
                          conv_out_ch=conv_out_ch, filter_sizes=[3, 4, 5])
    predictor.predict(test_iterator, batch_size=BATCH_SIZE)



def main_ensemble_test_postreply():
    '''Main function for testing ensemble model.
    '''
    EXPERIMENT_NAME_RNN = 'new_october'
    EXPERIMENT_NAME_CNN = 'new_october_CNN'
    BATCH_SIZE = 256

    params_RNN = open_experiment(EXPERIMENT_NAME_RNN)
    params_CNN = open_experiment(EXPERIMENT_NAME_CNN)
    cfg_path_RNN = params_RNN['cfg_path']
    cfg_path_CNN = params_CNN['cfg_path']
    vocab_size = params_RNN['Network']['vocab_size']
    PAD_IDX = params_RNN['Network']['PAD_IDX']
    UNK_IDX = params_RNN['Network']['UNK_IDX']
    classes = params_RNN['Network']['classes']
    MAX_VOCAB_SIZE = params_RNN['Network']['MAX_VOCAB_SIZE']
    SPLIT_RATIO = params_RNN['Network']['SPLIT_RATIO']
    EMBEDDING_DIM = params_RNN['Network']['EMBEDDING_DIM']
    HIDDEN_DIM = params_RNN['Network']['HIDDEN_DIM']
    conv_out_ch = params_CNN['Network']['conv_out_ch']
    MODEL_MODE = 'ensemble'
    pretrained_embeddings = torch.zeros((vocab_size, EMBEDDING_DIM))

    # Prepare data
    data_handler_test_RNN = data_provider_PostReply(cfg_path=cfg_path_RNN, batch_size=BATCH_SIZE, split_ratio=SPLIT_RATIO,
                                         max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TEST, model_mode='RNN')
    data_handler_test_CNN = data_provider_PostReply(cfg_path=cfg_path_CNN, batch_size=BATCH_SIZE, split_ratio=SPLIT_RATIO,
                                         max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TEST, model_mode="CNN")
    test_iterator_RNN = data_handler_test_RNN.data_loader()
    test_iterator_CNN = data_handler_test_CNN.data_loader()
    # Initialize predictor
    predictor = Prediction(cfg_path=params_RNN['cfg_path'], model_mode=MODEL_MODE, classes=classes,
                           cfg_path_RNN=cfg_path_RNN, cfg_path_CNN=cfg_path_CNN)

    predictor.setup_model(model=biLSTM, vocab_size=vocab_size, embeddings=pretrained_embeddings,
                          embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX,
                          conv_out_ch=conv_out_ch, filter_sizes=[3, 4, 5], model_c =CNN1d, model_r=biLSTM)
    predictor.predict_ensemble(test_iterator_RNN, test_iterator_CNN, batch_size=BATCH_SIZE)


def test_every_epoch():
    EXPERIMENT_NAME = 'new_october_CNN'
    BATCH_SIZE = 256

    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']
    vocab_size = params['Network']['vocab_size']
    PAD_IDX = params['Network']['PAD_IDX']
    UNK_IDX = params['Network']['UNK_IDX']
    classes = params['Network']['classes']
    MAX_VOCAB_SIZE = params['Network']['MAX_VOCAB_SIZE']
    SPLIT_RATIO = params['Network']['SPLIT_RATIO']
    EMBEDDING_DIM = params['Network']['EMBEDDING_DIM']
    HIDDEN_DIM = params['Network']['HIDDEN_DIM']
    conv_out_ch = params['Network']['conv_out_ch']
    MODEL_MODE = params['Network']['MODEL_MODE']
    pretrained_embeddings = torch.zeros((vocab_size, EMBEDDING_DIM))

    # Prepare data
    data_handler_test = data_provider_PostReply(cfg_path=cfg_path, batch_size=BATCH_SIZE, split_ratio=SPLIT_RATIO,
                                         max_vocab_size=MAX_VOCAB_SIZE, mode=Mode.TEST, model_mode=MODEL_MODE)
    test_iterator = data_handler_test.data_loader()
    # Initialize predictor
    predictor = Prediction(cfg_path, model_mode=MODEL_MODE, classes=classes)

    test_acc = pd.DataFrame(columns=['epoch', 'accuracy'])
    test_F1 = pd.DataFrame(columns=['epoch', 'F1'])
    for epoch in range(150):
        EPOCH = epoch + 1
        print('epoch:', EPOCH)

        if MODEL_MODE == "RNN":
            MODEL = biLSTM
        elif MODEL_MODE == "CNN":
            MODEL = CNN1d

        predictor.setup_model(model=MODEL, vocab_size=vocab_size, embeddings=pretrained_embeddings,
                              embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, pad_idx=PAD_IDX, unk_idx=UNK_IDX,
                              epoch=EPOCH, conv_out_ch=conv_out_ch, filter_sizes=[3,4,5])
        acc, F1 = predictor.predict(test_iterator, batch_size=BATCH_SIZE)

        test_acc = test_acc.append(pd.DataFrame([[EPOCH, acc]], columns=['epoch', 'accuracy']))
        test_F1 = test_F1.append(pd.DataFrame([[EPOCH, F1]], columns=['epoch', 'F1']))
        test_F1.to_csv(os.path.join(params['output_data_path'], 'test_F1.csv'), index=False)
        test_acc.to_csv(os.path.join(params['output_data_path'], 'test_acc.csv'), index=False)





def prediction_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



if __name__ == '__main__':
    # delete_experiment("new_october_CNN")
    # main_train()
    # main_test()
    # main_manual_predict(prediction_mode='Manualpart2')
    # main_reply_predict('philipp')
    # main_train_postreply()
    # main_test_postreply()
    # test_every_epoch()
    main_ensemble_test_postreply()