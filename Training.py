"""
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""


#System Modules
import os.path
from enum import Enum
import datetime

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
    Various functionalities in the training process such as setting up of devices, defining model and its parameters,
    executing training can be found here.
    '''
    def __init__(self, cfg_path, torch_seed=None):
        '''
        cfg_path (string):
            path of the experiment config file
        torch_seed (int):
            Seed used for random generators in PyTorch functions
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
        Setup the model by defining the model, optimiser,loss function and learning rate.

        :param model: an object of our network
        :param optimiser: an object of our optimizer, e.g. torch.optim.SGD
        :param optimiser_params: is a dictionary containing parameters for the optimiser, e.g. {'lr':7e-3}
        '''
        #Tensor Board Graph
        self.add_tensorboard_graph(model)

        self.model = model.to(self.device)
        self.optimiser = optimiser(self.model.parameters(), **optimiser_params)
        self.loss_function = loss_function(reduction='none')

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
        '''
        Creates a tensor board graph for network visualisation
        '''
        dummy_input = torch.rand(1, 256).long()  # To show tensor sizes in graph
        dummy_hidden = torch.rand(1, 256, 1024)  # To show tensor sizes in graph
        self.writer.add_graph(model, (dummy_input, dummy_hidden), verbose=False)


    def execute_training(self, train_loader, test_loader=None, num_epochs=None):
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
        self.step = 0
        print('Starting time:' + str(datetime.datetime.now()) +'\n')
        for epoch in range(num_epochs):
            self.epoch = epoch
            print('\nTraining:')
            self.train_epoch(train_loader)
            print('')
            if test_loader:
                print('Testing:')
                self.test_epoch(test_loader)

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
        print("Epoch [{}/{}] \n".format(self.epoch +1, self.model_info['num_epochs']))

        # loss value to display statistics
        total_loss = 0
        train_accuracy = 0
        batch_count = 0
        batch_accuracy = 0

        # initializing hidden states
        hidden_units = self.model.initialize_hidden_state(self.device)

        for batch, (message, label) in enumerate(train_loader):
            message = message.long()
            label = label.long()
            message = message.to(self.device)
            label = label.to(self.device)


            #Forward pass.
            self.optimiser.zero_grad()

            with torch.set_grad_enabled(True):
                output, hidden_unit = self.model(message.permute(1,0), hidden_units)

                # Loss & converting from one-hot encoding to class indices
                # pdb.set_trace()
                loss = self.loss_function(output, torch.max(label, 1)[1])
                loss = torch.mean(loss)
                total_loss += (loss / label.shape[1]).item()
                batch_count += 1

                # number of correct message predictions
                corrects = (torch.max(output, 1)[1].data == torch.max(label, 1)[1]).sum()
                # Accuracy
                batch_accuracy += 100.0 * corrects / len(output)

                #Backward and optimize
                loss.backward()
                self.optimiser.step()

                #TODO: metrics to be modified.

                # Prints loss statistics and writes to the tensorboard after number of steps specified.
                if (batch)%self.params['display_stats_freq'] == 0:
                    print('Epoch {} Batch {} Loss {}'.format(self.epoch + 1, batch, total_loss / batch_count))
                    self.calculate_tb_stats(total_loss / batch_count, batch_accuracy / batch_count)
                    total_loss = 0
                    batch_count = 0
                    train_accuracy += batch_accuracy
                    batch_accuracy = 0

        # Print accuracy after each epoch
        print('Epoch {} -- Train Acc. {}'.format(
            self.epoch + 1, train_accuracy / (batch+1) ))

    def test_epoch(self, test_loader):
        '''Test model after an epoch and calculate loss on test dataset'''
        self.model.eval()

        with torch.no_grad():
            # loss value to display statistics
            total_loss = 0
            test_accuracy = 0
            batch_count = 0
            batch_accuracy = 0

            hidden_units = self.model.initialize_hidden_state(self.device)

            for batch, (message, label) in enumerate(test_loader):
                message = message.long()
                label = label.long()
                message = message.to(self.device)
                label = label.to(self.device)

                output, hidden_units = self.model(message.permute(1,0), hidden_units)

                # Loss & converting from one-hot encoding to class indices
                loss = self.loss_function(output, torch.max(label, 1)[1])
                loss = torch.mean(loss)
                total_loss += (loss / label.shape[1]).item()
                batch_count += 1

                # number of correct message predictions
                corrects = (torch.max(output, 1)[1].data == torch.max(label, 1)[1]).sum()
                # Accuracy
                batch_accuracy += 100.0 * corrects / len(output)


                if batch % 5 == 0:
                    print('Epoch {} Batch {} Loss {}'.format(self.epoch + 1, batch, total_loss / batch_count))
                    self.calculate_tb_stats(total_loss / batch_count, batch_accuracy / batch_count)
                    total_loss = 0
                    batch_count = 0
                    test_accuracy += batch_accuracy
                    batch_accuracy = 0

        # Print accuracy after each epoch
        print('Epoch {} -- Test Acc. {}'.format(
            self.epoch + 1, test_accuracy / (batch+1) ))

        self.model.train()


    def calculate_tb_stats(self, batch_loss, batch_accuracy, is_train=True):
        '''
        Adds the statistics of metrics to the tensorboard.
        '''
        if is_train:
            mode='Training'
        else:
            mode='Validation'
            
        # Adds loss value & number & accuracy of correct predictions to TensorBoard
        self.writer.add_scalar(mode+'_Loss', batch_loss, self.step)
        self.writer.add_scalar(mode+'_Accuracy', batch_accuracy, self.step)

        # Adds all the network's trainable parameters to TensorBoard
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, self.step)
            self.writer.add_histogram(f'{name}.grad', param.grad, self.step)
        self.step += 1

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


class Mode(Enum):
    '''
    Class Enumerating the 3 modes of operation of the network.
    This is used while loading datasets
    '''

    TRAIN = 0
    VALID = 1
    PREDICT = 2


