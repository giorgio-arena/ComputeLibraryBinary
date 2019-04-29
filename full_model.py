import numpy as np
import tensorflow as tf
from keras import regularizers, optimizers, backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler

from binary_conv2d import BinaryConv2D
from accurate_conv2d import ABCConv2D


# Define constants
BATCH_SIZE = 64
EPOCHS     = 200


# Define learning rate
def lr_schedule(epoch):
    lrate = 0.001

    '''
    # Uncomment for XNOR
    if epoch > 25:
        lrate = 0.00001
    '''
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate


# Add convolution layer to model: if binary, bn -> act -> binary_conv; otherwise conv -> bn -> act
def add_conv_layer(model, out_channels, is_binary=False, binarize_input=False, dropout=None, in_shape=None):
    if is_binary:
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(BinaryConv2D(out_channels, (3, 3), binarize_input, dropout))
    else:
        if in_shape is None:
            model.add(Conv2D(out_channels, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
        else:
            model.add(Conv2D(out_channels, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4), input_shape=in_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    if dropout is not None:
        model.add(MaxPooling2D(pool_size=(2,2)))
        if not binarize_input:
            model.add(Dropout(dropout))


# Define model
def get_model(is_binary, binarize_input):
    assert(not binarize_input or is_binary)

    model = Sequential()

    add_conv_layer(model, 32, in_shape=x_train.shape[1:])
    add_conv_layer(model, 32, is_binary, binarize_input, 0.2)
    add_conv_layer(model, 64, is_binary, binarize_input)
    add_conv_layer(model, 64, is_binary, binarize_input, 0.3)
    add_conv_layer(model, 128, is_binary, binarize_input)
    add_conv_layer(model, 128, dropout=0.4)
    
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    return model


if __name__ == '__main__':
    
    # Load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    
    # Normalize data
    mean    = np.mean(x_train, axis=(0,1,2,3))
    std     = np.std(x_train, axis=(0,1,2,3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test  = (x_test - mean) / (std + 1e-7)
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test, 10)

    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,)
    datagen.fit(x_train)

    # Get models
    models_pairs = [('Full', get_model(False, False)), ('BinWeights', get_model(True, False)), ('XNOR', get_model(True, True))]


    # Train models
    for model_name, model in models_pairs:
        # Compile model using RMSprop optimizer
        opt_rms = optimizers.RMSprop(lr=0.001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])

        # Fit augmented data into the model and train
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), steps_per_epoch=x_train.shape[0] // BATCH_SIZE, epochs=EPOCHS,
                            verbose=2, validation_data=(x_test,y_test), callbacks=[LearningRateScheduler(lr_schedule)])
       
        # Save to disk as numpy arrays
        counters = [0, 0]
        for l in model.layers:
            if(type(l) == BatchNormalization):
                layer_name = "BatchNorm" + str(counters[0])
                weights_names = ["Gamma", "Beta", "Mean", "Std"]
                counters[0] += 1
            elif(type(l) == Conv2D or type(l) == BinaryConv2D):
                layer_name = "Conv" + str(counters[1])
                weights_names = ["Weights", "Bias"]
                counters[1] += 1
            else:
                continue  # Only save BatchNorm and Conv

            filename = model_name + layer_name
            weights  = l.get_weights()
            
            if(type(l) == BinaryConv2D):
                del weights[1]  # Discard approx_weights

            for w, n in zip(weights, weights_names):
                np.save(filename + n, w)

        # Test
        scores = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
        print('\n' + model_name + ' validation, accuracy: %.3f loss: %.3f\n\n' % (scores[1] * 100, scores[0]))