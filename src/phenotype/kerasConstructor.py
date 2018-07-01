import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import np_utils
from keras.utils import plot_model

#use convolutional neural networks
from keras.layers import Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator


from keras.datasets import mnist
from keras.datasets import cifar10
import sys

from src import parameters
from sklearn.model_selection import StratifiedShuffleSplit
from keras import backend as K


class KerasConstructor(object):

    def __init__(self, root):
        sess = tf.Session()
        K.set_session(sess)
        temp = self.createModel(root)
        self.testAcc = temp[1]
        self.trainAcc = temp[0]

    def createModel(self, pheno):
        phenoArr = pheno[0]
        order = pheno[1]

        # ----------- NETWORK PARAMETERS -----------
        # default activation function is RELU
        actFuncExit = parameters.ACTIVATION_FUNCTION_FOR_EXIT
        optimizer = parameters.OPTIMIZER
        lossFunc = parameters.LOSS_FUNCTION
        # ----------- MODEL FIT PARAMETERS - MNIST -----------
        # import paramater values
        inpDime = parameters.INPUT_DIMENSION
        outDime = parameters.OUTPUT_DIMENSION
        num_epoch = parameters.LEARN_EPOCH_COUNT
        batch_size = parameters.BATCH_SIZE
        num_class = parameters.OUTPUT_CLASS_COUNT
        VERBOSE = parameters.VERBOSE
        isConvolution = parameters.USE_CONVOLUTION_NN

        seed = 1337
        n_iter = 1
        train_size = parameters.TRAIN_SIZE

        # ----------- PREPARE TEST & TRAIN DATASET -----------
        if parameters.DATASET == 'MNIST':
            (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
            if train_size < 60000:
                sss = StratifiedShuffleSplit(n_splits=n_iter, test_size=parameters.OUTPUT_DIMENSION, train_size=train_size, random_state=seed)
                sss.get_n_splits(X_train, Y_train)
                for train_index, test_index in sss.split(X_train, Y_train):
                    X_train, Y_train = X_train[train_index], Y_train[train_index]
            if isConvolution:
                X_train = X_train.reshape(train_size, inpDime[0],inpDime[0],inpDime[1])
                X_test = X_test.reshape(10000, inpDime[0],inpDime[0],inpDime[1])
                X_train = X_train.astype('float32')
                X_test = X_test.astype('float32')
                X_train /= 255
                X_test /= 255
                Y_train = np_utils.to_categorical(Y_train, num_class)
                Y_test = np_utils.to_categorical(Y_test, num_class)
            else :
                X_train = X_train.reshape(train_size, inpDime)
                X_test = X_test.reshape(10000, inpDime)
                X_train = X_train.astype('float32')
                X_test = X_test.astype('float32')
                X_train /= 255
                X_test /= 255
                Y_train = np_utils.to_categorical(Y_train, num_class)
                Y_test = np_utils.to_categorical(Y_test, num_class)

        if parameters.DATASET == 'CIFAR':
            (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
            if train_size < 50000:
                sss = StratifiedShuffleSplit(n_splits=n_iter, test_size=parameters.OUTPUT_DIMENSION, train_size=train_size, random_state=seed)
                sss.get_n_splits(X_train, Y_train)
                for train_index, test_index in sss.split(X_train, Y_train):
                    X_train, Y_train = X_train[train_index], Y_train[train_index]

            if isConvolution:
                X_train = X_train.reshape(train_size, inpDime[0],inpDime[0],inpDime[1])
                X_test = X_test.reshape(10000, inpDime[0],inpDime[0],inpDime[1])
                X_train = X_train.astype('float32')
                X_test = X_test.astype('float32')
                X_train /= 255
                X_test /= 255
                Y_train = np_utils.to_categorical(Y_train, num_class)
                Y_test = np_utils.to_categorical(Y_test, num_class)
            else :
                X_train = X_train.reshape(train_size, inpDime)
                X_test = X_test.reshape(10000, inpDime)
                X_train = X_train.astype('float32')
                X_test = X_test.astype('float32')
                X_train /= 255
                X_test /= 255
                Y_train = np_utils.to_categorical(Y_train, num_class)
                Y_test = np_utils.to_categorical(Y_test, num_class)



        # ----------- PREPARE NETWORK ----------------------------------------------------------------------------------
        modelArr = [None]*len(phenoArr)
        if isConvolution:
            modelArr[0] = (Input(shape=(inpDime[0],inpDime[0],inpDime[1])))
        else:
            modelArr[0] = (Input(shape=(inpDime,)))
        output_layers = []
        # --- LAYERS
        index = 0
        safety = 3*len(order)
        while len(order) > 0:
            if safety < 0:
                print('ERROR: KERAS BUILD MODEL LOOPING')
                return 0
            safety -= 1
            index = index % len(order)
            order_index = order[index]
            layer = phenoArr[order_index] # iterate through pheno nodes
            # for layers with more than 1 input concatenate all previously created layers and use the concatenations
            # as input for newly created layer
            isReady = True
            for inp in layer.inputs:
                if modelArr[inp.index] is None :
                    isReady = False
                    break

            if isReady:
                if len(layer.inputs) > 1:
                    layersToConcatenate = []
                    for inp in layer.inputs:
                        layersToConcatenate.append(modelArr[inp.index])
                    x = keras.layers.concatenate(layersToConcatenate)
                else :
                    for inp in layer.inputs:
                        x = (modelArr[inp.index])
                        # for layers with only one input create new layer and use the layer
                if not isConvolution:
                    modelArr[order_index] = Dense(layer.neuron_count, activation=layer.act_func)(x)
                else:
                    im_dim = parameters.IMG_DIMENSION
                    filters = layer.filter_count
                    kernel_size = layer.kernel_size if layer.kernel_size < im_dim else im_dim
                    pool_size = layer.pool_size if layer.pool_size < im_dim else im_dim
                    dropout = layer.dropout
                    x = modelArr[inp.index]
                    x = Conv2D(filters = filters, kernel_size = kernel_size,
                               strides=1, padding='same', activation = layer.act_func)(x)
                    if layer.maxPooling:
                        x = MaxPooling2D(pool_size = (pool_size, pool_size),
                                         strides=1, padding='same')(x)
                    if layer.dropout > 0:
                        Dropout(dropout)(x)
                    modelArr[order_index] = x

                if len(layer.outputs) == 0: # mark all ouput layers
                    output_layers.append(modelArr[order_index])
                order.remove(order_index)
                index = 0
            else :
                index += 1
                # -----
        # ----------- CREATE NETWORK -----------
        input_layer = modelArr[0]
        if len(output_layers) > 1:
            x = keras.layers.concatenate(output_layers)
        else:
            x = output_layers[0]
        if isConvolution:
            x = Flatten()(x)
            BatchNormalization()(x)
            x = Dense(parameters.MIN_NEURON_THRESHOLD, activation=actFuncExit)(x)
            x = Dropout(0.5)(x)
        output_layer = Dense(outDime, activation=actFuncExit)(x)

        # --------------------------------------------------------------------------------------------------------------
        # ----------- MODEL EVALUATE-----------
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=lossFunc, optimizer=optimizer, metrics=['accuracy'])
        model.fit(X_train, Y_train,
                  batch_size = batch_size,
                  epochs = num_epoch,
                  verbose = VERBOSE,
                  validation_data = (X_test, Y_test))

        # ----------- NETWORK OUTPUT ACCURACY-----------
        score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
        testScore = score[1]*100
        score = model.evaluate(X_train, Y_train, verbose=VERBOSE)
        trainScore = score[1]*100
        return trainScore,testScore

    def printScore(self, test, train):
        outp = ' ---- TRAIN: ' + str(train*100) + ' ------ TEST: ' + str(test*100)
        file = open(parameters.OUTPUT_ACCURACY, 'a')
        file.write('\n' + outp)
        file.close()