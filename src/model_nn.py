import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense

from keras.layers import Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D

from keras.datasets import mnist
from keras.datasets import cifar10

from src.convnetdrawer.convnet_drawer import Model as drawModel, drawConv2D, drawMaxPooling2D, drawFlatten, drawDense
from src.convnetdrawer.matplotlib_util import save_model_to_file

from src import constants, plotNN
from sklearn.model_selection import StratifiedShuffleSplit
from keras import backend as K


class ModelNN(object):

    def __init__(self, phenotype, no_of_net):
        sess = tf.Session()
        K.set_session(sess)
        nn = self.create_model(phenotype, no_of_net)
        self.train_acc = nn[0]
        self.test_acc = nn[1]

    def create_model(self, phenotype, no_of_net):
        phenotypes = phenotype[0]
        order = phenotype[1]

        seed = 42  # the meaning of life
        n_splits = 1

        if constants.DATASET == 'MNIST':
            (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
            if constants.TRAIN_SIZE < 60000:
                sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=constants.OUTPUT_DIMENSION,
                                             train_size=constants.TRAIN_SIZE, random_state=seed)
                sss.get_n_splits(X_train, Y_train)
                for train_index, test_index in sss.split(X_train, Y_train):
                    X_train, Y_train = X_train[train_index], Y_train[train_index]
        elif constants.DATASET == 'CIFAR10':
            (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
            if constants.TRAIN_SIZE < 50000:
                sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=constants.OUTPUT_DIMENSION,
                                             train_size=constants.TRAIN_SIZE, random_state=seed)
                sss.get_n_splits(X_train, Y_train)
                for train_index, test_index in sss.split(X_train, Y_train):
                    X_train, Y_train = X_train[train_index], Y_train[train_index]

        if constants.USE_CNN:
            X_train = X_train.reshape(constants.TRAIN_SIZE, constants.INPUT_DIMENSION[0], constants.INPUT_DIMENSION[0],
                                      constants.INPUT_DIMENSION[1])
            X_test = X_test.reshape(constants.TEST_SIZE, constants.INPUT_DIMENSION[0], constants.INPUT_DIMENSION[0],
                                    constants.INPUT_DIMENSION[1])
        else:
            X_train = X_train.reshape(constants.TRAIN_SIZE, constants.INPUT_DIMENSION)
            X_test = X_test.reshape(constants.TEST_SIZE, constants.INPUT_DIMENSION)
        X_train = X_train.astype('float32')
        X_train /= 255
        Y_train = np_utils.to_categorical(Y_train, constants.K_CLASS_COUNT)
        X_test = X_test.astype('float32')
        X_test /= 255
        Y_test = np_utils.to_categorical(Y_test, constants.K_CLASS_COUNT)

        layers = [None] * len(phenotypes)  # preinit necessary cause of direct access based on order
        list_of_layers = []
        if constants.USE_CNN:
            layers[0] = (Input(
                shape=(constants.INPUT_DIMENSION[0], constants.INPUT_DIMENSION[0], constants.INPUT_DIMENSION[1])))
            model = drawModel(input_shape=(
                constants.INPUT_DIMENSION[0], constants.INPUT_DIMENSION[0], constants.INPUT_DIMENSION[1]))
        else:
            layers[0] = (Input(shape=(constants.INPUT_DIMENSION,)))
            list_of_layers.append((round(constants.INPUT_DIMENSION / 100), 'relu'))
        output_layers = []

        index = 0
        while len(order) > 0:
            order_index = order[index]  # always use the first element
            layer = phenotypes[order_index]  # iterate through Phenotype nodes
            # for layers with more than 1 input concatenate all previously created layers and use the concatenations
            # as input for newly created layer

            inputs_compiled = True
            for input in layer.inputs:
                inputs_compiled &= (layers[input.index] is not None)

            if inputs_compiled:
                if len(layer.inputs) > 1:
                    concatenation = []
                    for input in layer.inputs:
                        concatenation.append(layers[input.index])
                    x = keras.layers.concatenate(concatenation)
                else:
                    for input in layer.inputs:
                        # for layers with only one input create new layer and use the layer
                        x = (layers[input.index])
                if not constants.USE_CNN:
                    layers[order_index] = Dense(layer.neurons, activation=layer.activation_function)(x)
                    list_of_layers.append((round(layer.neurons / 100), layer.activation_function))
                else:
                    x = Conv2D(filters=layer.filter_count, kernel_size=layer.kernel_size,
                               strides=(1, 1), padding='same', activation=layer.activation_function)(x)
                    model.add(drawConv2D(filters=layer.filter_count, kernel_size=(layer.kernel_size, layer.kernel_size),
                                         strides=(1, 1), padding='same'))
                    if layer.maxPooling:
                        x = MaxPooling2D(pool_size=(layer.pool_size, layer.pool_size),
                                         strides=(1, 1), padding='same')(x)
                        model.add(drawMaxPooling2D(pool_size=(layer.pool_size, layer.pool_size),
                                                   strides=(1, 1), padding='same'))

                    Dropout(layer.dropout)(x)
                    # model.add(drawDropout(layer.dropout))
                    layers[order_index] = x

                if len(layer.outputs) == 0:  # mark all output layers
                    output_layers.append(layers[order_index])
                order.remove(order_index)
                index = 0
            else:
                index += 1

        input_layer = layers[0]
        if len(output_layers) > 1:
            x = keras.layers.concatenate(output_layers)
        else:
            x = output_layers[0]
        if constants.USE_CNN:
            x = Flatten()(x)
            model.add(drawFlatten())
            BatchNormalization()(x)
            # model.add(drawBatchNormalization())
            x = Dense(constants.MIN_NEURONS, activation=constants.K_ACTIVATION_FUNCTION_OUTPUT_LAYER)(x)
            model.add(drawDense(constants.MIN_NEURONS))
            x = Dropout(0.5)(x)
            # model.add(Dropout(0.5))
        output_layer = Dense(constants.OUTPUT_DIMENSION, activation=constants.K_ACTIVATION_FUNCTION_OUTPUT_LAYER)(x)
        if constants.USE_CNN:
            model.add(drawDense(constants.OUTPUT_DIMENSION))
            # save_model_to_file(model, "CNN" + str(no_of_net) + ".pdf")
        else:
            list_of_layers.append((round(constants.OUTPUT_DIMENSION), constants.K_ACTIVATION_FUNCTION_OUTPUT_LAYER))
            # TODO: only draw best of each generation in a file
            # plotNN.DrawNN(list_of_layers).draw()

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=constants.K_LOSS, optimizer=constants.K_OPTIMIZER, metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=constants.BATCH_SIZE, epochs=constants.K_EPOCHS,
                  verbose=constants.K_VERBOSE, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=constants.K_VERBOSE)
        test_score = score[1] * 100
        score = model.evaluate(X_train, Y_train, verbose=constants.K_VERBOSE)
        train_score = score[1] * 100
        return train_score, test_score, list_of_layers
