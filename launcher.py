'''
DL models (FNN, 1D CNN and CNN-LSTM) evaluation on N-CMAPSS
12.07.2021
Hyunho Mo
hyunho.mo@unitn.it
'''
## Import libraries in python
import gc

import os
import json
import logging
import sys
import h5py
import time
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
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
from tqdm import tqdm
import scipy.stats as stats
# from sklearn.utils.testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning
# import keras
import tensorflow as tf
print(tf.__version__)
# import keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.data_preparation import df_all_creator, df_train_creator, df_test_creator, Input_Gen

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
seed = 0
random.seed(0)
np.random.seed(seed)
# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')




def main():

    # Load data
    '''
    W: operative conditions (Scenario descriptors)
    X_s: measured signals
    X_v: virtual sensors
    T(theta): engine health parameters
    Y: RUL [in cycles]
    A: auxiliary data
    '''

    df_all = df_all_creator(data_filepath)

    '''
    Split dataframe into Train and Test
    Training units: 2, 5, 10, 16, 18, 20
    Test units: 11, 14, 15

    '''
    # units = list(np.unique(df_A['unit']))
    units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
    units_index_test = [11.0, 14.0, 15.0]

    print("units_index_train", units_index_train)
    print("units_index_test", units_index_test)



    df_train = df_train_creator(df_all, units_index_train)
    df_test = df_test_creator(df_all, units_index_test)

    print(df_train)
    print(df_train.columns)
    print("num of inputs: ", len(df_train.columns) )
    print(df_test)
    print(df_test.columns)
    print("num of inputs: ", len(df_test.columns))


    del df_all
    gc.collect()
    df_all = pd.DataFrame()



    cols_normalize = df_train.columns.difference(['RUL', 'unit'])
    sequence_length = 50
    sequence_cols = df_train.columns.difference(['RUL', 'unit'])
    print('check0')
    data_class = Input_Gen (df_train, df_test, cols_normalize, sequence_length, sequence_cols)
    print ('check1')
    train_samples, label_array_train, test_samples, truth_array_test = data_class.seq_gen()
    print('check2')
    print("truth_array_test.shape", train_samples.shape)
    print("truth_array_test.shape", label_array_train.shape)
    print("truth_array_test.shape", test_samples.shape)
    print("truth_array_test.shape", truth_array_test.shape)






if __name__ == '__main__':
    main()
