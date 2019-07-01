import pandas as pd
from sklearn import linear_model
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import os


class Learn:
    def __init__(self, data_frame=[]):
        self.__data_frame = data_frame
        d = {'a': self.__data_frame['a'], 'b': self.__data_frame['b'], 'c': self.__data_frame['c'],
             'd': self.__data_frame['d']}
        __x__ = pd.DataFrame(data=d)
        __y__ = self.__data_frame['y']

    def __learn__(self):
        print('Learn is pure virtual function')


class LRLearn(Learn):
    def __init__(self, data_frame=[]):
        print('LRLearn is instantiated')
        self.__data_frame = data_frame
        d = {'a': self.__data_frame['a'], 'b': self.__data_frame['b'], 'c': self.__data_frame['c'],
             'd': self.__data_frame['d']}
        self.__x__ = pd.DataFrame(data=d)
        self.__y__ = self.__data_frame['y']

    def __learn__(self):
        reg = linear_model.LinearRegression()
        reg.fit(self.__x__, self.__y__)
        intercept = reg.intercept_
        coefficients = reg.coef_

        print('Intercept : ', intercept)
        print('Coefficients : ', coefficients)

        h5f = h5py.File(os.path.abspath('output/linear_reg_model.hdf5'), 'w')
        h5f.create_dataset('intercept', data=np.array(intercept))
        h5f.create_dataset('coefficients', data=np.array(coefficients))
        h5f.close()

        # save the model to models folder - copy/manual action of move
        h5f1 = h5py.File(os.path.abspath('models/linear_reg_model.hdf5'), 'w')
        h5f1.create_dataset('intercept', data=np.array(intercept))
        h5f1.create_dataset('coefficients', data=np.array(coefficients))
        h5f1.close()


class NNLearn(Learn):
    def __init__(self, data_frame=[]):
        print('NNLearn is instantiated')
        self.__data_frame = data_frame
        d = {'a': self.__data_frame['a'], 'b': self.__data_frame['b'], 'c': self.__data_frame['c'],
             'd': self.__data_frame['d']}
        self.__x__ = pd.DataFrame(data=d)
        self.__y__ = self.__data_frame['y']

    def __learn__(self):
        model = Sequential()
        model.add(Dense(1, kernel_initializer='uniform', activation='relu', input_dim=4))
        sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        model.fit(self.__x__, self.__y__, epochs=12)

        model.save(filepath=os.path.abspath('output/neural_networks_model.h5'))
        model.save_weights(os.path.abspath('output/nn_model_weights.h5'))

        model.save(filepath=os.path.abspath('models/neural_networks_model.h5'))
        model.save_weights(os.path.abspath('models/nn_model_weights.h5'))

        del model

        print("Saved NN model to models folder")
