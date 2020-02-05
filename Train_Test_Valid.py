"""
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""

#System Modules
import os.path
from enum import Enum
import datetime
import time
import spacy

# Deep Learning Modules
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from sklearn import metrics

# User Defined Modules
from configs.serde import *
import pdb
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class Training:
    '''
    This class represents training process.
    '''
    def __init__(self, cfg_path, num_epochs=10, RESUME=False, torch_seed=None):
        '''
        :cfg_path (string): path of the experiment config file
        :torch_seed (int): Seed used for random generators in PyTorch functions
        '''
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.RESUME = RESUME
        self.best_loss = float('inf')

        if RESUME == False:
            self.model_info = self.params['Network']
            self.model_info['seed'] = torch_seed or self.model_info['seed']
            self.epoch = 0
            self.num_epochs = num_epochs

            if 'trained_time' in self.model_info:
                self.raise_training_complete_exception()

            self.setup_cuda()
            self.writer = SummaryWriter(log_dir=os.path.join(self.params['tb_logs_path']))


    def setup_cuda(self, cuda_device_id=0):
        '''Setup the CUDA device'''
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
            torch.cuda.manual_seed_all(self.model_info['seed'])
            torch.manual_seed(self.model_info['seed'])
        else:
            self.device = torch.device('cpu')


    def setup_model(self, model, optimiser, optimiser_params, loss_function):
        '''
        :param model: an object of our network
        :param optimiser: an object of our optimizer, e.g. torch.optim.SGD
        :param optimiser_params: is a dictionary containing parameters for the optimiser, e.g. {'lr':7e-3}
        '''
        # number of parameters of the model
        print(f'\nThe model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters!\n')

        #Tensor Board Graph
        self.add_tensorboard_graph(model)

        self.model = model.to(self.device)
        self.optimiser = optimiser(self.model.parameters(), **optimiser_params)
        self.loss_function = loss_function()

        if 'retrain' in self.model_info and self.model_info['retrain']==True:
            self.load_pretrained_model()

        # Saves the model, optimiser,loss function name for writing to config file
        # self.model_info['model_name'] = model.__name__
        self.model_info['optimiser'] = optimiser.__name__
        self.model_info['loss_function'] = loss_function.__name__
        self.model_info['optimiser_params'] = optimiser_params
        self.params['Network']=self.model_info
        write_config(self.params, self.cfg_path,sort_keys=True)


    def load_checkpoint(self, model, optimiser, optimiser_params, loss_function):

        checkpoint = torch.load(self.params['network_output_path'] + '/' + self.params['checkpoint_name'])
        self.device = None
        self.model_info = checkpoint['model_info']
        self.setup_cuda()
        self.model = model.to(self.device)
        self.optimiser = optimiser(self.model.parameters(), **optimiser_params)
        self.loss_function = loss_function()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.num_epochs = checkpoint['num_epoch']
        self.loss_function = checkpoint['loss']
        self.best_loss = checkpoint['best_loss']
        self.writer = SummaryWriter(log_dir=os.path.join(self.params['tb_logs_path']), purge_step=self.epoch + 1)


    def add_tensorboard_graph(self, model):
        '''Creates a tensor board graph for network visualisation'''
        dummy_input = torch.rand(19, 1).long()  # To show tensor sizes in graph
        dummy_text_length = torch.ones(1).long()  # To show tensor sizes in graph
        self.writer.add_graph(model, (dummy_input, dummy_text_length))


    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


    def execute_training(self, train_loader, valid_loader=None, batch_size=1):
        '''
        Executes training by running training and validation at each epoch
        '''
        total_start_time = time.time()

        if self.RESUME == False:
            # Checks if already trained
            if 'trained_time' in self.model_info:
                self.raise_training_complete_exception

            # CODE FOR CONFIG FILE TO RECORD MODEL PARAMETERS
            self.model_info = self.params['Network']
            self.model_info['num_epochs'] = self.num_epochs or self.model_info['num_epochs']

        print('Starting time:' + str(datetime.datetime.now()) +'\n')

        for epoch in range(self.num_epochs - self.epoch):
            self.epoch += 1
            start_time = time.time()

            print('Training (intermediate metrics):')
            train_loss, train_acc, train_F1 = self.train_epoch(train_loader, batch_size)

            if valid_loader:
                print('\nValidation (intermediate metrics):')
                valid_loss, valid_acc, valid_F1 = self.valid_epoch(valid_loader, batch_size)

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            total_mins, total_secs = self.epoch_time(total_start_time, end_time)

            # Writes to the tensorboard after number of steps specified.
            if valid_loader:
                self.calculate_tb_stats(train_loss, train_F1, valid_loss, valid_F1)
            else:
                self.calculate_tb_stats(train_loss, train_F1)

            # Saves information about training to config file
            self.model_info['num_steps'] = self.epoch
            self.model_info['trained_time'] = "{:%B %d, %Y, %H:%M:%S}".format(datetime.datetime.now())
            self.params['Network'] = self.model_info
            write_config(self.params, self.cfg_path, sort_keys=True)

            '''Saving the model'''
            if valid_loader:
                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    torch.save(self.model.state_dict(), self.params['network_output_path'] + '/' +
                               self.params['trained_model_name'])
            else:
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    torch.save(self.model.state_dict(), self.params['network_output_path'] + '/' +
                               self.params['trained_model_name'])

            # Saving every 20 epochs
            if (self.epoch) % self.params['network_save_freq'] == 0:
                torch.save(self.model.state_dict(), self.params['network_output_path'] + '/' +
                           'epoch{}_'.format(self.epoch) + self.params['trained_model_name'])

            # Save a checkpoint
            torch.save({'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimiser.state_dict(),
                'loss': self.loss_function, 'num_epoch': self.num_epochs,
                'model_info': self.model_info, 'best_loss': self.best_loss},
                self.params['network_output_path'] + '/' + self.params['checkpoint_name'])

            # Print accuracy, F1, and loss after each epoch
            print('\n---------------------------------------------------------------')
            print(f'Epoch: {self.epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | '
                  f'Total Time so far: {total_mins}m {total_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train F1: {train_F1:.3f}')
            if valid_loader:
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% |  Val. F1: {valid_F1:.3f}')
            print('---------------------------------------------------------------\n')


    def train_epoch(self, train_loader, batch_size):
        '''
        Train using one single iteration of all messages (epoch) in dataset
        '''
        print("Epoch [{}/{}]".format(self.epoch, self.model_info['num_epochs']))
        self.model.train()
        previous_idx = 0

        # initializing the loss list
        batch_loss = 0
        batch_count = 0

        # initializing the caches
        logits_cache = torch.from_numpy(np.zeros((len(train_loader) * batch_size, 3)))
        max_preds_cache = torch.from_numpy(np.zeros((len(train_loader) * batch_size, 1)))
        labels_cache = torch.from_numpy(np.zeros(len(train_loader) * batch_size))

        for idx, batch in enumerate(train_loader):
            message, message_lengths = batch.text
            label = batch.label
            message = message.long()
            label = label.long()
            message = message.to(self.device)
            label = label.to(self.device)

            #Forward pass.
            self.optimiser.zero_grad()

            with torch.set_grad_enabled(True):
                output = self.model(message, message_lengths).squeeze(1)

                # Loss
                loss = self.loss_function(output, label)
                batch_loss += loss.item()
                batch_count += 1
                max_preds = output.argmax(dim=1, keepdim=True)  # get the index of the max probability

                # saving the logits and labels of this batch
                for i, batch_vector in enumerate(max_preds):
                    max_preds_cache[idx * batch_size + i] = batch_vector
                for i, batch_vector in enumerate(output):
                    logits_cache[idx * batch_size + i] = batch_vector
                for i, value in enumerate(label):
                    labels_cache[idx * batch_size + i] = value

                #Backward and optimize
                loss.backward()
                self.optimiser.step()

                # Prints loss statistics after number of steps specified.
                if (idx + 1)%self.params['display_stats_freq'] == 0:
                    print('Epoch {:02} | Batch {:03}-{:03} | Train loss: {:.3f}'.
                          format(self.epoch, previous_idx, idx, batch_loss / batch_count))
                    previous_idx = idx + 1
                    batch_loss = 0
                    batch_count = 0

        '''Metrics calculation over the whole set'''
        # Accuracy
        correct = max_preds_cache.squeeze(1).eq(labels_cache)
        epoch_accuracy = (correct.sum() / torch.FloatTensor([labels_cache.shape[0]])).item()

        max_preds_cache = max_preds_cache.cpu()
        labels_cache = labels_cache.cpu()

        # F1 Score
        epoch_f1_score = metrics.f1_score(labels_cache, max_preds_cache, average='macro')
        labels_cache = labels_cache.long()

        # Loss
        loss = self.loss_function(logits_cache.to(self.device), labels_cache.to(self.device))
        epoch_loss = loss.item()

        return epoch_loss, epoch_accuracy, epoch_f1_score


    def valid_epoch(self, valid_loader, batch_size):
        '''Test (validation) model after an epoch and calculate loss on valid dataset'''
        print("Epoch [{}/{}]".format(self.epoch, self.model_info['num_epochs']))
        self.model.eval()
        previous_idx = 0

        with torch.no_grad():
            # initializing the loss list
            batch_loss = 0
            batch_count = 0

            # initializing the caches
            logits_cache = torch.from_numpy(np.zeros((len(valid_loader) * batch_size, 3)))
            max_preds_cache = torch.from_numpy(np.zeros((len(valid_loader) * batch_size, 1)))
            labels_cache = torch.from_numpy(np.zeros(len(valid_loader) * batch_size))

            for idx, batch in enumerate(valid_loader):
                message, message_lengths = batch.text
                label = batch.label
                message = message.long()
                label = label.long()
                message = message.to(self.device)
                label = label.to(self.device)
                output = self.model(message, message_lengths).squeeze(1)

                # Loss
                loss = self.loss_function(output, label)
                batch_loss += loss.item()
                batch_count += 1
                max_preds = output.argmax(dim=1, keepdim=True)  # get the index of the max probability

                # saving the logits and labels of this batch
                for i, batch_vector in enumerate(max_preds):
                    max_preds_cache[idx * batch_size + i] = batch_vector
                for i, batch_vector in enumerate(output):
                    logits_cache[idx * batch_size + i] = batch_vector
                for i, value in enumerate(label):
                    labels_cache[idx * batch_size + i] = value

                # Prints loss statistics after number of steps specified.
                if (idx + 1)%self.params['display_stats_freq'] == 0:
                    print('Epoch {:02} | Batch {:03}-{:03} | Val. loss: {:.3f}'.
                          format(self.epoch, previous_idx, idx, batch_loss / batch_count))
                    previous_idx = idx + 1
                    batch_loss = 0
                    batch_count = 0

        '''Metrics calculation over the whole set'''
        # Accuracy
        correct = max_preds_cache.squeeze(1).eq(labels_cache)
        epoch_accuracy = (correct.sum() / torch.FloatTensor([labels_cache.shape[0]])).item()

        max_preds_cache = max_preds_cache.cpu()
        labels_cache = labels_cache.cpu()

        # F1 Score
        epoch_f1_score = metrics.f1_score(labels_cache, max_preds_cache, average='macro')
        labels_cache = labels_cache.long()

        # Loss
        loss = self.loss_function(logits_cache.to(self.device), labels_cache.to(self.device))
        epoch_loss = loss.item()

        self.model.train()
        return epoch_loss, epoch_accuracy, epoch_f1_score


    def calculate_tb_stats(self, train_loss, train_F1, valid_loss=None, valid_F1=None):
        '''Adds the statistics of metrics to the tensorboard'''

        # Adds the metrics to TensorBoard
        self.writer.add_scalar('Training' + '_Loss', train_loss, self.epoch)
        self.writer.add_scalar('Training' + '_F1', train_F1, self.epoch)
        if valid_loss:
            self.writer.add_scalar('Validation' + '_Loss', valid_loss, self.epoch)
            self.writer.add_scalar('Validation' + '_F1', valid_F1, self.epoch)

        # Adds all the network's trainable parameters to TensorBoard
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, self.epoch)
            self.writer.add_histogram(f'{name}.grad', param.grad, self.epoch)


    def load_pretrained_model(self):
        '''Load pre trained model to the using pre-trained_model_path parameter from config file'''
        self.model.load_state_dict(torch.load(self.model_info['pretrain_model_path']))
        
    def raise_training_complete_exception(self):
        raise Exception("Model has already been trained on {}. \n"
                                            "1.To use this model as pre trained model and train again\n "
                                            "create new experiment using create_retrain_experiment function.\n\n"
                                            "2.To start fresh with same experiment name, delete the experiment  \n"
                                            "using delete_experiment function and create experiment "
                            "               again.".format(self.model_info['trained_time']))



class Prediction:
    '''
    This class represents prediction (testing) process similar to the Training class.
    '''
    def __init__(self, cfg_path):
        '''
        :cfg_path (string): path of the experiment config file
        '''
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.setup_cuda()


    def setup_cuda(self, cuda_device_id=0):
        '''Setup the CUDA device'''
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


    def setup_model(self, model, vocab_size, embeddings, pad_idx, unk_idx, model_file_name=None):
        '''
        Setup the model by defining the model, load the model from the pth file saved during training.
        '''
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']
        self.model_p = model(vocab_size=vocab_size, embeddings=embeddings,
                             pad_idx=pad_idx, unk_idx=unk_idx).to(self.device)

        # Loads model from model_file_name and default network_output_path
        self.model_p.load_state_dict(torch.load(self.params['network_output_path'] + "/" + model_file_name))


    def predict(self, test_loader, batch_size):
        # Reads params to check if any params have been changed by user
        self.params = read_config(self.cfg_path)
        self.model_p.eval()

        start_time = time.time()
        with torch.no_grad():
            # initializing the caches
            logits_cache = torch.from_numpy(np.zeros((len(test_loader) * batch_size, 3)))
            max_preds_cache = torch.from_numpy(np.zeros((len(test_loader) * batch_size, 1)))
            labels_cache = torch.from_numpy(np.zeros(len(test_loader) * batch_size))

            for idx, batch in enumerate(test_loader):
                message, message_lengths = batch.text
                label = batch.label
                message = message.long()
                label = label.long()
                message = message.to(self.device)
                label = label.to(self.device)
                output = self.model_p(message, message_lengths).squeeze(1)
                max_preds = output.argmax(dim=1, keepdim=True)  # get the index of the max probability

                # saving the logits and labels of this batch
                for i, batch_vector in enumerate(max_preds):
                    max_preds_cache[idx * batch_size + i] = batch_vector
                for i, batch_vector in enumerate(output):
                    logits_cache[idx * batch_size + i] = batch_vector
                for i, value in enumerate(label):
                    labels_cache[idx * batch_size + i] = value

        '''Metrics calculation over the whole set'''
        # Accuracy
        correct = max_preds_cache.squeeze(1).eq(labels_cache)
        final_accuracy = (correct.sum() / torch.FloatTensor([labels_cache.shape[0]])).item()

        max_preds_cache = max_preds_cache.cpu()
        labels_cache = labels_cache.cpu()

        # F1 Score
        final_f1_score = metrics.f1_score(labels_cache, max_preds_cache, average='macro')

        end_time = time.time()
        test_mins, test_secs = self.epoch_time(start_time, end_time)

        # Print the final accuracy and F1 score
        print('\n----------------------------------------------------------------------')
        print(f'Testing for the SemEval 2014 and 2015 gold data | Testing Time: {test_mins}m {test_secs}s')
        print(f'\tTesting Acc: {final_accuracy * 100:.2f}% | Testing F1: {final_f1_score:.3f}')
        print('----------------------------------------------------------------------\n')


    def manual_predict(self, labels, vocab_idx, phrase, min_len = 4, tokenizer=spacy.load('en'), mode=None):
        '''
        Manually predicts the polarity of the given sentence.
        Possible polarities: 1.neutral, 2.positive, 3.negative
        '''
        # Reads params to check if any params have been changed by user
        self.params = read_config(self.cfg_path)
        self.model_p.eval()

        tokenized = [tok.text for tok in tokenizer.tokenizer(phrase)]
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [vocab_idx[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(1)
        preds = self.model_p(tensor, torch.Tensor([tensor.shape[0]]))
        max_preds = preds.argmax(dim=1)

        if mode == Mode.REPLY_PREDICTION:
            return labels[max_preds.item()]

        print('\n----------------------------------')
        print(f'\tThis is a {labels[max_preds.item()]} phrase!')
        print('----------------------------------')



class Mode(Enum):
    '''
    Class Enumerating the 3 modes of operation of the network.
    This is used while loading datasets
    '''
    TRAIN = 0
    VALID = 1
    TEST = 2
    PREDICTION = 3


