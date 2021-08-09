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




def one_dcnn(n_filters, kernel_size, input_array):

    cnn = Sequential(name='one_d_cnn')
    cnn.add(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', input_shape=(input_array.shape[1],input_array.shape[2])))
    # cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same'))
    # cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv1D(filters=1, kernel_size=kernel_size, padding='same'))
    # cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Flatten())
    print(cnn.summary())

    return cnn