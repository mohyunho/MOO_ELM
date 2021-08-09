



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