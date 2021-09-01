#!/bin/python3
"""
This file contains the implementation of a Task, used to load the data and compute the fitness of an individual

"""
import pandas as pd
from abc import abstractmethod

# from input_creator import input_gen
from elm_network import network_fit

class Task:
    @abstractmethod
    def get_n_parameters(self):
        pass

    @abstractmethod
    def get_parameters_bounds(self):
        pass

    @abstractmethod
    def evaluate(self, genotype):
        pass


class SimpleNeuroEvolutionTask(Task):
    '''
    TODO: Consider hyperparameters of ELM instead of the number of neurons in hidden layers of MLPs.
    Class for EA Task
    '''
    def __init__(self, train_sample_array, train_label_array, val_sample_array, val_label_array, model_path, device):
        self.train_sample_array = train_sample_array
        self.train_label_array = train_label_array
        self.val_sample_array = val_sample_array
        self.val_label_array = val_label_array
        self.model_path = model_path
        self.device = device

    def get_n_parameters(self):
        return 13

    def get_parameters_bounds(self):
        bounds = [
            (1, 5), #L2 norm params, 0
            (1, 5), #numb of hidden layers, 1
            (10, 500), #hidden1 neurons, 2
            (10, 100), #hidden2 neurons, 3
            (10, 100), #hidden3 neurons, 4
            (10, 100), #hidden4 neurons, 5
            (10, 100), #hidden5 neurons, 6
            (1, 5),  # hidden1 type, 7
            (1, 5),  # hidden2 type, 8
            (1, 5),  # hidden3 type, 9
            (1, 5),  # hidden4 type, 10
            (1, 5),  # hidden5 type, 11
            (1, 5) #activation, 12
        ]
        return bounds

    def evaluate(self, genotype):
        '''
        Create input & generate NNs & calculate fitness (to evaluate fitness of each individual)
        :param genotype:
        :return:
        '''
        l2_parms_lst = [1, 0.1, 0.01, 0.001, 0.0001]
        l2_parm = l2_parms_lst(genotype[0]-1)
        num_layer = genotype[1]
        type_cand_lst = ["lin", "sigm", "tanh", "rbf_l2", "rbf_linf"]

        num_neuron_lst = []
        type_neuron_lst = []

        for i in range(5):
            num_neuron_lst.append(0)
            type_neuron_lst.append("none")


        for n in range(num_layer):
            num_neuron_lst[n] = genotype[n+2]*10
            type_neuron_lst[n] = type_cand_lst [genotype[n+7]-1]


        elm_net = network_fit(self.train_sample_array, self.train_label_array,
                              self.val_sample_array, self.val_label_array,
                              l2_parm, num_layer,
                              num_neuron_lst, type_neuron_lst, self.model_path, self.device)



        fitness = elm_net.train_net(epochs=self.epochs, batch_size=self.batch)


        return fitness

