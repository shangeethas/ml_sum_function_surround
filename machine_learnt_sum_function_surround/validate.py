import h5py
import os
from keras.models import load_model
import pandas as pd



class Validate:
    def __init__(self, data_frame=[]):
        self.__data_frame = data_frame

    def load_nn_model_architecture_1(self):
        print('Validate Neural Networks Model 1')
        model = load_model(os.path.abspath('models/neural_networks_model_1.h5'))
        model.load_weights(os.path.abspath('models/nn_model_weights_1.h5'))
        return model

    def load_nn_model_architecture_2(self):
        print('Validate Neural Networks Model 2')
        model = load_model(os.path.abspath('models/neural_networks_model_2.h5'))
        model.load_weights(os.path.abspath('models/nn_model_weights_2.h5'))
        return model

    def validate_regression_model(self):
        print('Validate Regression Model')
        h5f = h5py.File(os.path.abspath('models/linear_reg_model.hdf5'), 'r')
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

    def validate_neural_networks_model(self):

        d = {'a': self.__data_frame['a'], 'b': self.__data_frame['b'], 'c': self.__data_frame['c'],
             'd': self.__data_frame['d']}
        _x_ = pd.DataFrame(data=d)
        _y_ = self.__data_frame['y']

        model1 = self.load_nn_model_architecture_1()

        score, accuracy = model1.evaluate(_x_, _y_)

        print('Score of Neural Networks Model 1 : ', score)
        print('Accuracy of Neural Networks Model 1 :', accuracy)

        model2 = self.load_nn_model_architecture_2()

        score, accuracy = model2.evaluate(_x_, _y_)

        print('Score of Neural Networks Model 2 : ', score)
        print('Accuracy of Neural Networks Model 2 :', accuracy)