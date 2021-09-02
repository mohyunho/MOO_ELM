import time
import json
import logging as log
import sys

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform

import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error

from math import sqrt
# import keras
np.random.seed(0)

from utils.hpelm import ELM, HPELM


def score_calculator(y_predicted, y_actual):
    # Score metric
    h_array = y_predicted - y_actual
    s_array = np.zeros(len(h_array))
    print ("calculating score")
    for j, h_j in enumerate(h_array):
        if h_j < 0:
            s_array[j] = math.exp(-(h_j / 13)) - 1

        else:
            s_array[j] = math.exp(h_j / 10) - 1
    score = np.sum(s_array)
    return score


def gen_net(train_sample_array, l2_norm, num_layer, num_neurons_lst, type_lst, device = "GPU"):
    '''
    Generate and evaluate any ELM
    :param
    :return:
    '''

    model = HPELM(train_sample_array.shape[1], 1, accelerator=device, batch=1000, norm=l2_norm)
    for idx in range(num_layer):
        model.add_neurons(num_neurons_lst[idx], type_lst[idx])

    return model


class network_fit(object):
    '''
    class for network
    '''

    def __init__(self, train_sample_array, train_label_array, val_sample_array, val_label_array,
                 l2_parm, num_layer, num_neurons_lst, type_lst, model_path, device):
        '''
        Constructor
        Generate a NN and train
        @param none
        '''
        # self.__logger = logging.getLogger('data preparation for using it as the network input')
        self.train_sample_array = train_sample_array
        self.train_label_array = train_label_array
        self.val_sample_array = val_sample_array
        self.val_label_array = val_label_array
        self.l2_parm = l2_parm
        self.num_layer = num_layer
        self.num_neurons_lst = num_neurons_lst
        self.type_lst = type_lst
        self.model_path = model_path
        self.device = device


        self.model= gen_net(self.train_sample_array, self.l2_parm, self.num_layer,
                            self.num_neurons_lst, self.type_lst, self.device)



    def train_net(self, batch_size= 1000):
        '''
        specify the optimizers and train the network
        :param epochs:
        :param batch_size:
        :param lr:
        :return:
        '''
        print("Initializing network...")
        start_itr = time.time()
        elm = self.model
        elm.train(self.train_sample_array, self.train_label_array, "r")
        print ("individual trained...evaluation in progress...")


        pred_test = elm.predict(self.val_sample_array)

        print ("prediction completed")



        h_array = pred_test - self.val_label_array
        print ("h_array")
        s_array = np.zeros(len(h_array))
        print("calculating score")
        for j, h_j in enumerate(h_array):
            if h_j < 0:
                s_array[j] = math.exp(-(h_j / 13)) - 1

            else:
                s_array[j] = math.exp(h_j / 10) - 1
        score = np.sum(s_array)


        print("score: ", score)


        rms = sqrt(mean_squared_error(pred_test, self.val_label_array))
        # print(rms)
        rms = round(rms, 4)
        fitness_net = (rms,)
        end_itr = time.time()
        print("training network is successfully completed, time: ", end_itr - start_itr)

        print("fitness in rmse: ", fitness_net[0])

        return fitness_net




    #
    #
    #
    # def test_net(self, epochs = 1000, batch_size= 700, lr= 1e-05, plotting=True):
    #     '''
    #     Evalute the trained network on test set
    #     :param trained_net:
    #     :param best_model:
    #     :param plotting:
    #     :return:
    #     '''
    #     print("Test, Initializing network...")
    #     start_itr = time.time()
    #     # compile the model
    #     # mlps.compile(optimizer='adam', loss='mse')
    #     rp = optimizers.RMSprop(learning_rate=lr, rho=0.9, centered=True)
    #     adm = optimizers.Adam(learning_rate=lr, epsilon=1)
    #     sgd_m = optimizers.SGD(learning_rate=lr)
    #
    #     keras_rmse = tf.keras.metrics.RootMeanSquaredError()
    #     # mlps.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[ keras_rmse,'mae'])
    #     # mlps.compile(loss='mean_squared_error', optimizer=rp, metrics=[ keras_rmse,'mae'])
    #     self.mlps.compile(loss='mean_squared_error', optimizer=sgd_m, metrics=[keras_rmse, 'mae'])
    #     # mlps.compile(loss='mean_squared_error', optimizer=adm, metrics=[ keras_rmse,'mae'])
    #     # print(self.mlps.summary())
    #     # Train the model
    #     # history = mlps.fit(rp_train_samples, rp_train_label, epochs=epochs, batch_size=batch_size, verbose=1)
    #     history = self.mlps.fit(self.train_samples, self.label_array_train, epochs=epochs, batch_size=batch_size,
    #                             validation_split=0.2, verbose=2,
    #                             callbacks=[
    #                            EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=2, mode='min'),
    #                            ModelCheckpoint(self.model_path, monitor='val_root_mean_squared_error', save_best_only=True, mode='min',
    #                                            verbose=2)])
    #
    #     val_rmse_k = history.history['val_root_mean_squared_error']
    #     val_rmse_min = min(val_rmse_k)
    #     min_val_rmse_idx = val_rmse_k.index(min(val_rmse_k))
    #     stop_epoch = min_val_rmse_idx +1
    #
    #     val_rmse_min = round(val_rmse_min, 4)
    #     fitness_net = (val_rmse_min,)
    #
    #     end_itr = time.time()
    #
    #     print("training network is successfully completed, time: ", end_itr - start_itr)
    #     print("min_val_rmse: ", fitness_net[0])
    #
    #     estimator = load_model(self.model_path)
    #
    #     y_pred_test = estimator.predict(self.test_samples)
    #     y_true_test = self.label_array_test
    #
    #     pd.set_option('display.max_rows', 1000)
    #     test_print = pd.DataFrame()
    #     test_print['y_pred'] = y_pred_test.flatten()
    #     test_print['y_truth'] = y_true_test.flatten()
    #     test_print['diff'] = abs(y_pred_test.flatten() - y_true_test.flatten())
    #     test_print['diff(ratio)'] = abs(y_pred_test.flatten() - y_true_test.flatten()) / y_true_test.flatten()
    #     test_print['diff(%)'] = (abs(y_pred_test.flatten() - y_true_test.flatten()) / y_true_test.flatten()) * 100
    #     # print(test_print)
    #
    #     y_predicted = test_print['y_pred']
    #     y_actual = test_print['y_truth']
    #     rms = sqrt(mean_squared_error(y_actual, y_predicted)) # RMSE metric
    #     test_print['rmse'] = rms
    #     print(test_print)
    #
    #
    #     # Score metric
    #     h_array = y_predicted - y_actual
    #     s_array = np.zeros(len(h_array))
    #     for j, h_j in enumerate(h_array):
    #         if h_j < 0:
    #             s_array[j] = math.exp(-(h_j / 13)) - 1
    #
    #         else:
    #             s_array[j] = math.exp(h_j / 10) - 1
    #     score = np.sum(s_array)
    #
    #     # Plot the results of RUL prediction
    #     if plotting == True:
    #         fig_verify = plt.figure(figsize=(12, 6))
    #         plt.plot(y_pred_test, color="blue")
    #         plt.plot(y_true_test, color="green")
    #         plt.title('prediction')
    #         plt.ylabel('value')
    #         plt.xlabel('row')
    #         plt.legend(['predicted', 'actual data'], loc='upper left')
    #         plt.show()
    #
    #     return rms, score
