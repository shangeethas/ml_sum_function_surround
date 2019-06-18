import h5py
import os
from keras.models import Sequential
from keras.layers import Dense


class Validate:
    def __init__(self, data_frame=[]):
        self.__data_frame = data_frame

    def __define_nn_model_architecture__(self):
        model = Sequential()
        model.add(Dense(1, kernel_initializer='uniform', activation='relu', input_shape=(4, )))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def __validate_regression_model__(self):
        print('Validate Regression Model : ', os.path.abspath('models/linear_reg_model_1.hdf5'))
        h5f = h5py.File(os.path.abspath('models/linear_reg_model_1.hdf5'), 'r')
        intercept = h5f['intercept'][()]
        coefficients = h5f['coefficients'][()]
        h5f.close()

        sum_error = 0.0
        for i in range(self.__data_frame['a'].count() - 1):
            y_model = coefficients[0] * self.__data_frame['a'][i] + coefficients[1] * self.__data_frame['b'][i] + \
                      coefficients[2] * self.__data_frame['c'][i] + coefficients[3] * self.__data_frame['d'][i] + \
                      intercept
            y_actual = self.__data_frame['y'][i]

            error = y_model - y_actual
            sum_error += (error ** 2)

        print('Mean Sum Error of Linear Regression Model : ', sum_error / self.__data_frame['a'].count())

    def __validate_neural_networks_model__(self):
        model = self.__define_nn_model_architecture__()
        model.load_weights(filepath=os.path.abspath('models/neural_networks_model.hdf5'))
        print("Loaded model weights from disk")

        score, accuracy = model.evaluate((self.__data_frame['a'], self.__data_frame['b'],
                                          self.__data_frame['c'], self.__data_frame['d']),
                                         self.__data_frame['y'])

        print('Score of Neural Networks Model : ', score)
        print('Accuracy of Neural Networks Model :', accuracy)
