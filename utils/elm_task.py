#!/bin/python3
"""
This file contains the implementation of a Task, used to load the data and compute the fitness of an individual

"""
import pandas as pd
from abc import abstractmethod

# from input_creator import input_gen
from utils.elm_network import network_fit

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
    def __init__(self, train_sample_array, train_label_array, val_sample_array, val_label_array, batch, model_path, device):
        self.train_sample_array = train_sample_array
        self.train_label_array = train_label_array
        self.val_sample_array = val_sample_array
        self.val_label_array = val_label_array
        self.batch = batch
        self.model_path = model_path
        self.device = device

    def get_n_parameters(self):
        return 6

    def get_parameters_bounds(self):
        bounds = [
            (1, 5), #L2 norm params, 0
            (10, 200), #type1 neurons, 1
            (10, 200), #type2 neurons, 2
            (10, 200), #type3 neurons, 3
            (10, 200), #type4 neurons, 4
            (10, 200), #type5 neurons, 5
        ]
        return bounds

    def evaluate(self, genotype):
        '''
        Create input & generate NNs & calculate fitness (to evaluate fitness of each individual)
        :param genotype:
        :return:
        '''
        print ("######################################################################################")
        l2_parms_lst = [1, 0.1, 0.01, 0.001, 0.0001]
        l2_parm = l2_parms_lst[genotype[0]-1]
        type_neuron_lst = ["lin", "sigm", "tanh", "rbf_l2", "rbf_linf"]

        num_neuron_lst = []

        for n in range(5):
            num_neuron_lst[n] = genotype[n+1]*10


        print("l2_params: " ,l2_parm)
        print("num_neuron_lst: ", num_neuron_lst)
        print("type_neuron_lst: ", type_neuron_lst)



        elm_net = network_fit(self.train_sample_array, self.train_label_array,
                              self.val_sample_array, self.val_label_array,
                              l2_parm,
                              num_neuron_lst, type_neuron_lst, self.model_path, self.device)


        fitness = elm_net.train_net(batch_size=self.batch)

        return fitness

