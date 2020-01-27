"""
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""


#System Modules
import os.path
from enum import Enum
import datetime
import time

# Deep Learning Modules
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torch.nn as nn

# User Defined Modules
from configs.serde import *
import pdb


class Training:
    '''
    This class represents training process.
    '''
    def __init__(self, cfg_path, torch_seed=None):
        '''
        :cfg_path (string): path of the experiment config file
        :torch_seed (int): Seed used for random generators in PyTorch functions
        '''
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.model_info = self.params['Network']
        self.model_info['seed'] = torch_seed or self.model_info['seed']

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


    def execute_training(self, train_loader, valid_loader=None, num_epochs=None):
        '''
        Executes training by running training and validation at each epoch
        '''
        # reads param file again to include changes if any
        self.params = read_config(self.cfg_path)

        # Checks if already trained
        if 'trained_time' in self.model_info:
            self.raise_training_complete_exception

        # CODE FOR CONFIG FILE TO RECORD MODEL PARAMETERS
        self.model_info = self.params['Network']
        self.model_info['num_epochs'] = num_epochs or self.model_info['num_epochs']

        self.epoch = 0
        self.tb_train_step = 0
        self.tb_val_step = 0
        print('Starting time:' + str(datetime.datetime.now()) +'\n')
        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()
            print('Training:')
            train_loss, train_acc = self.train_epoch(train_loader)
            if valid_loader:
                print('\nValidation:')
                valid_loss, valid_acc = self.valid_epoch(valid_loader)

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            # Print accuracy and loss after each epoch
            print('\n-----------------------------------------------')
            print(f'Epoch: {self.epoch + 1} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            if valid_loader:
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
            print('-----------------------------------------------\n')

        '''Saving the model'''
            # Saving every epoch
            # torch.save(self.model.state_dict(), self.params['network_output_path'] +
            #            "/epoch_" + str(self.epoch) + '_' + self.params['trained_model_name'])
        # Saving the last epoch
        # torch.save(self.model.state_dict(), self.params['network_output_path'] +
        # "/" + self.params['trained_model_name'])

        # Saves information about training to config file
        self.model_info['num_steps'] = self.epoch
        self.model_info['trained_time'] = "{:%B %d, %Y, %H:%M:%S}".format(datetime.datetime.now())
        self.params['Network'] = self.model_info

        write_config(self.params, self.cfg_path, sort_keys=True)



    def train_epoch(self, train_loader):
        '''
        Train using one single iteration of all messages (epoch) in dataset
        '''
        print("Epoch [{}/{}]".format(self.epoch +1, self.model_info['num_epochs']))
        self.model.train()
        previous_idx = 0

        # loss value to display statistics
        total_loss = 0
        batch_loss = 0
        total_accuracy = 0
        batch_count = 0
        batch_accuracy = 0

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
                total_loss += loss.item()
                batch_count += 1

                # Accuracy
                max_preds = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
                correct = max_preds.squeeze(1).eq(label)
                batch_accuracy += (correct.sum() / torch.FloatTensor([label.shape[0]])).item()
                total_accuracy += (correct.sum() / torch.FloatTensor([label.shape[0]])).item()

                #Backward and optimize
                loss.backward()
                self.optimiser.step()

                #TODO: other metrics to be added.

                # Prints loss statistics and writes to the tensorboard after number of steps specified.
                if (idx + 1)%self.params['display_stats_freq'] == 0:
                    print('Epoch {} | Batch {}-{} | Average train loss: {:.3f}'.
                          format(self.epoch + 1, previous_idx, idx, batch_loss / batch_count))
                    previous_idx = idx + 1
                    self.tb_train_step += 1
                    self.calculate_tb_stats(batch_loss / batch_count, batch_accuracy / batch_count, is_train=True)
                    batch_loss = 0
                    batch_count = 0
                    batch_accuracy = 0

        epoch_accuracy = total_accuracy / len(train_loader)
        epoch_loss = total_loss / len(train_loader)
        return epoch_loss, epoch_accuracy


    def valid_epoch(self, valid_loader):
        '''Test (validation) model after an epoch and calculate loss on test dataset'''
        print("Epoch [{}/{}]".format(self.epoch + 1, self.model_info['num_epochs']))
        self.model.eval()
        previous_idx = 0

        with torch.no_grad():
            # loss value to display statistics
            total_loss = 0
            batch_loss = 0
            total_accuracy = 0
            batch_count = 0
            batch_accuracy = 0

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
                total_loss += loss.item()
                batch_count += 1

                # Accuracy
                max_preds = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
                correct = max_preds.squeeze(1).eq(label)
                batch_accuracy += (correct.sum() / torch.FloatTensor([label.shape[0]])).item()
                total_accuracy += (correct.sum() / torch.FloatTensor([label.shape[0]])).item()

                # Prints loss statistics and writes to the tensorboard after number of steps specified.
                if (idx + 1) % self.params['display_stats_freq'] == 0:
                    print('Epoch {} | Batch {}-{} | Average Val. loss: {:.3f}'.
                          format(self.epoch + 1, previous_idx, idx, batch_loss / batch_count))
                    previous_idx = idx + 1
                    self.tb_val_step += 1
                    self.calculate_tb_stats(batch_loss / batch_count, batch_accuracy / batch_count, is_train=False)
                    batch_loss = 0
                    batch_count = 0
                    batch_accuracy = 0

        self.model.train()
        epoch_accuracy = total_accuracy / len(valid_loader)
        epoch_loss = total_loss / len(valid_loader)
        return epoch_loss, epoch_accuracy


    def calculate_tb_stats(self, batch_loss, batch_accuracy, is_train=True):
        '''Adds the statistics of metrics to the tensorboard'''
        if is_train:
            mode='Training'
            step = self.tb_train_step
        else:
            mode='Validation'
            step = self.tb_val_step
            
        # Adds loss value & number & accuracy of correct predictions to TensorBoard
        self.writer.add_scalar(mode+'_Loss', batch_loss, step)
        self.writer.add_scalar(mode+'_Accuracy', batch_accuracy, step)

        # Adds all the network's trainable parameters to TensorBoard
        if is_train:
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param, self.tb_train_step)
                self.writer.add_histogram(f'{name}.grad', param.grad, self.tb_train_step)


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
    TO BE IMPLEMENTED.
    '''
    pass



class Mode(Enum):
    '''
    Class Enumerating the 3 modes of operation of the network.
    This is used while loading datasets
    '''

    TRAIN = 0
    VALID = 1
    PREDICT = 2


