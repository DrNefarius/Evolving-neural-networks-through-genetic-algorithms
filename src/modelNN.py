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

from src import parameters, plotNN
from sklearn.model_selection import StratifiedShuffleSplit
from keras import backend as K


class ModelNN(object):

    def __init__(self, root, no_of_net):
        sess = tf.Session()
        K.set_session(sess)
        temp = self.create_model(root, no_of_net)
        self.test_acc = temp[1]
        self.train_acc = temp[0]

    def create_model(self, pheno, no_of_net):
        pheno_arr = pheno[0]
        order = pheno[1]

        seed = 42
        n_iter = 1
        train_size = parameters.TRAIN_SIZE

        if parameters.DATASET == 'MNIST':
            (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
            if train_size < 60000:
                sss = StratifiedShuffleSplit(n_splits=n_iter, test_size=parameters.OUTPUT_DIMENSION,
                                             train_size=train_size, random_state=seed)
                sss.get_n_splits(X_train, Y_train)
                for train_index, test_index in sss.split(X_train, Y_train):
                    X_train, Y_train = X_train[train_index], Y_train[train_index]
            if parameters.USE_CONVOLUTION_NN:
                X_train = X_train.reshape(train_size, parameters.INPUT_DIMENSION[0], parameters.INPUT_DIMENSION[0],
                                          parameters.INPUT_DIMENSION[1])
                X_train = X_train.astype('float32')
                X_train /= 255
                Y_train = np_utils.to_categorical(Y_train, parameters.OUTPUT_CLASS_COUNT)
                X_test = X_test.reshape(10000, parameters.INPUT_DIMENSION[0], parameters.INPUT_DIMENSION[0],
                                        parameters.INPUT_DIMENSION[1])
                X_test = X_test.astype('float32')
                X_test /= 255
                Y_test = np_utils.to_categorical(Y_test, parameters.OUTPUT_CLASS_COUNT)
            else:
                X_train = X_train.reshape(train_size, parameters.INPUT_DIMENSION)
                X_train = X_train.astype('float32')
                X_train /= 255
                Y_train = np_utils.to_categorical(Y_train, parameters.OUTPUT_CLASS_COUNT)
                X_test = X_test.reshape(10000, parameters.INPUT_DIMENSION)
                X_test = X_test.astype('float32')
                X_test /= 255
                Y_test = np_utils.to_categorical(Y_test, parameters.OUTPUT_CLASS_COUNT)

        if parameters.DATASET == 'CIFAR':
            (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
            if train_size < 50000:
                sss = StratifiedShuffleSplit(n_splits=n_iter, test_size=parameters.OUTPUT_DIMENSION,
                                             train_size=train_size, random_state=seed)
                sss.get_n_splits(X_train, Y_train)
                for train_index, test_index in sss.split(X_train, Y_train):
                    X_train, Y_train = X_train[train_index], Y_train[train_index]

            if parameters.USE_CONVOLUTION_NN:
                X_train = X_train.reshape(train_size, parameters.INPUT_DIMENSION[0], parameters.INPUT_DIMENSION[0],
                                          parameters.INPUT_DIMENSION[1])
                X_test = X_test.reshape(10000, parameters.INPUT_DIMENSION[0], parameters.INPUT_DIMENSION[0],
                                        parameters.INPUT_DIMENSION[1])
                X_train = X_train.astype('float32')
                X_test = X_test.astype('float32')
                X_train /= 255
                X_test /= 255
                Y_train = np_utils.to_categorical(Y_train, parameters.OUTPUT_CLASS_COUNT)
                Y_test = np_utils.to_categorical(Y_test, parameters.OUTPUT_CLASS_COUNT)
            else:
                X_train = X_train.reshape(train_size, parameters.INPUT_DIMENSION)
                X_test = X_test.reshape(10000, parameters.INPUT_DIMENSION)
                X_train = X_train.astype('float32')
                X_test = X_test.astype('float32')
                X_train /= 255
                X_test /= 255
                Y_train = np_utils.to_categorical(Y_train, parameters.OUTPUT_CLASS_COUNT)
                Y_test = np_utils.to_categorical(Y_test, parameters.OUTPUT_CLASS_COUNT)

        # ----------- PREPARE NETWORK ----------------------------------------------------------------------------------
        model_arr = [None] * len(pheno_arr)
        list_of_layers = []
        if parameters.USE_CONVOLUTION_NN:
            model_arr[0] = (Input(
                shape=(parameters.INPUT_DIMENSION[0], parameters.INPUT_DIMENSION[0], parameters.INPUT_DIMENSION[1])))
            model = drawModel(input_shape=(
                parameters.INPUT_DIMENSION[0], parameters.INPUT_DIMENSION[0], parameters.INPUT_DIMENSION[1]))
        else:
            model_arr[0] = (Input(shape=(parameters.INPUT_DIMENSION,)))
            list_of_layers.append((round(parameters.INPUT_DIMENSION / 100), 'relu'))
        output_layers = []

        index = 0
        while len(order) > 0:
            index = index % len(order)
            order_index = order[index]
            layer = pheno_arr[order_index]  # iterate through pheno nodes
            # for layers with more than 1 input concatenate all previously created layers and use the concatenations
            # as input for newly created layer
            is_ready = True
            for inp in layer.inputs:
                if model_arr[inp.index] is None:
                    is_ready = False
                    break

            if is_ready:
                if len(layer.inputs) > 1:
                    layers_to_concatenate = []
                    for inp in layer.inputs:
                        layers_to_concatenate.append(model_arr[inp.index])
                    x = keras.layers.concatenate(layers_to_concatenate)
                else:
                    for inp in layer.inputs:
                        x = (model_arr[inp.index])
                        # for layers with only one input create new layer and use the layer
                if not parameters.USE_CONVOLUTION_NN:
                    model_arr[order_index] = Dense(layer.neuron_count, activation=layer.activation_function)(x)
                    list_of_layers.append((round(layer.neuron_count / 100), layer.activation_function))
                else:
                    im_dim = parameters.IMG_DIMENSION
                    kernel_size = layer.kernel_size if layer.kernel_size < im_dim else im_dim
                    pool_size = layer.pool_size if layer.pool_size < im_dim else im_dim
                    dropout = layer.dropout
                    x = model_arr[inp.index]
                    x = Conv2D(filters=layer.filter_count, kernel_size=kernel_size,
                               strides=1, padding='same', activation=layer.activation_function)(x)
                    model.add(drawConv2D(filters=layer.filter_count, kernel_size=(kernel_size, kernel_size),
                                         strides=(1, 1), padding='same'))
                    if layer.maxPooling:
                        x = MaxPooling2D(pool_size=(pool_size, pool_size),
                                         strides=1, padding='same')(x)
                        model.add(drawMaxPooling2D(pool_size=(pool_size, pool_size),
                                                   strides=(1, 1), padding='same'))

                    if layer.dropout > 0:
                        Dropout(dropout)(x)
                        # model.add(drawDropout(dropout))
                    model_arr[order_index] = x

                if len(layer.outputs) == 0:  # mark all ouput layers
                    output_layers.append(model_arr[order_index])
                order.remove(order_index)
                index = 0
            else:
                index += 1
                # -----
        # ----------- CREATE NETWORK -----------
        input_layer = model_arr[0]
        if len(output_layers) > 1:
            x = keras.layers.concatenate(output_layers)
        else:
            x = output_layers[0]
        if parameters.USE_CONVOLUTION_NN:
            x = Flatten()(x)
            model.add(drawFlatten())
            BatchNormalization()(x)
            # model.add(drawBatchNormalization())
            x = Dense(parameters.MIN_NEURON_THRESHOLD, activation=parameters.ACTIVATION_FUNCTION_FOR_EXIT)(x)
            model.add(drawDense(parameters.MIN_NEURON_THRESHOLD))
            x = Dropout(0.5)(x)
            # model.add(Dropout(0.5))
        output_layer = Dense(parameters.OUTPUT_DIMENSION, activation=parameters.ACTIVATION_FUNCTION_FOR_EXIT)(x)
        if parameters.USE_CONVOLUTION_NN:
            model.add(drawDense(parameters.OUTPUT_DIMENSION))
            save_model_to_file(model, "CNN" + str(no_of_net) + ".pdf")
        else:
            list_of_layers.append((round(parameters.OUTPUT_DIMENSION), parameters.ACTIVATION_FUNCTION_FOR_EXIT))
            # TODO: only draw best of each generation in a file
            # plotNN.DrawNN(list_of_layers).draw()

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=parameters.LOSS_FUNCTION, optimizer=parameters.OPTIMIZER, metrics=['accuracy'])
        model.fit(X_train, Y_train,
                  batch_size=parameters.BATCH_SIZE,
                  epochs=parameters.LEARN_EPOCH_COUNT,
                  verbose=parameters.VERBOSE,
                  validation_data=(X_test, Y_test))

        score = model.evaluate(X_test, Y_test, verbose=parameters.VERBOSE)
        testScore = score[1] * 100
        score = model.evaluate(X_train, Y_train, verbose=parameters.VERBOSE)
        trainScore = score[1] * 100
        return trainScore, testScore, list_of_layers
