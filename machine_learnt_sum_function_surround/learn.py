import pandas as pd
from sklearn import linear_model
import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import os


class LRLearn:
    def __init__(self, data_frame=[]):
        self.__data_frame = data_frame

    def __learn_linear_regression__(self):
        d = {'a': self.__data_frame['a'], 'b': self.__data_frame['b'], 'c': self.__data_frame['c'], 'd': self.__data_frame['d']}
        x = pd.DataFrame(data=d)

        y = self.__data_frame['y']
        reg = linear_model.LinearRegression()
        reg.fit(x, y)
        intercept = reg.intercept_
        coefficients = reg.coef_

        print('Intercept : ', intercept)
        print('Coefficients : ', coefficients)

        # h5f = h5py.File(os.path.abspath('output/linear_reg_model.hdf5'), 'w')
        # h5f.create_dataset('intercept', data=np.array(intercept))
        # h5f.create_dataset('coefficients', data=np.array(coefficients))
        # h5f.close()

        # save the model to models folder - copy/manual action of move
        h5f1 = h5py.File(os.path.abspath('models/linear_reg_model_1.hdf5'), 'w')
        h5f1.create_dataset('intercept', data=np.array(intercept))
        h5f1.create_dataset('coefficients', data=np.array(coefficients))
        h5f1.close()


class NNLearn:
    def __init__(self, data_frame=[]):
        self.data_frame = data_frame

    def __learn_neural_networks__(self):
        model = Sequential()
        model.add(Dense(1, kernel_initializer='uniform', activation='relu', input_shape=(4, )))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.save(os.path.abspath('output/neural_networks_model.hdf5'))
        # model.save_weights(os.path.abspath('output/linear_reg_model.hdf5'))

        model.save(os.path.abspath('models/neural_networks_model.hdf5'))
        # model.save_weights(os.path.abspath('models/linear_reg_model.hdf5'))

        model_json = model.to_json()

        with open(os.path.abspath('output/neural_networks_model.hdf5'), "w") as json_file:
            json_file.write(model_json)

        print("Saved NN model to output folder")

        with open(os.path.abspath('models/neural_networks_model.hdf5'), "w") as json_file:
            json_file.write(model_json)

        print("Saved NN model to models folder")