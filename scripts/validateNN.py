import h5py
import os
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np


class Validate:
    def __init__(self, data_frame=[]):
        self.__data_frame = data_frame

    def __define_nn_model_architecture__(self):
        model = Sequential()
        model.add(Dense(1, kernel_initializer='uniform', activation='relu', input_shape=(4, )))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def __save_nn_model__(self):
        model = self.__define_nn_model_architecture__()
        model.save(os.path.abspath('../models/neural_networks_model_test_1.hdf5'))

    def __validate_neural_networks_model__(self):
        model = self.__define_nn_model_architecture__()
        # model.load_weights(filepath=os.path.abspath('../models/neural_networks_model_test_1.hdf5'))
        print("Loaded model weights from disk")

        x = {'a': self.__data_frame['a'], 'b': self.__data_frame['b'],
             'c': self.__data_frame['c'], 'd': self.__data_frame['d']}
        x_test = pd.DataFrame(x)

        print('Data frame X Test values : ', x_test.values)

        y = {'y': self.__data_frame['y']}
        y_test = pd.DataFrame(y)

        print('Data frame Y Test values : ', y_test.values)

        score, accuracy = model.evaluate(np.array(x_test.values), np.array(y_test.values))

        print('Score of Neural Networks Model : ', score)
        print('Accuracy of Neural Networks Model :', accuracy)


if __name__ == '__main__':
    data_frame = pd.read_csv('../data/numbers_test.csv')
    validator = Validate(data_frame)
    validator.__save_nn_model__()
    validator.__validate_neural_networks_model__()

