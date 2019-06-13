import h5py
import os

class Validate:
    def __init__(self, data_frame=[]):
        self.__data_frame = data_frame

    def __validate_regression_model__(self):
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

    def __validate_neural_networks_model__(self):
        # h5f = h5py.File(os.path.abspath('models/neural_networks_model.hdf5'), 'r')
        # h5f.load_weights(os.path.abspath('models/neural_networks_model.hdf5'))
        # print("Loaded model from disk")
        #
        # h5f.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        sum_score = 0.0

        # for i in range(self.__data_frame['a'].count() - 1):
        #     score = h5f.evaluate((self.__data_frame['a'][i], self.__data_frame['b'][i], self.__data_frame['c'][i],
        #                           self.__data_frame['d'][i]), self.__data_frame['y'][i], verbose=0)
        #     sum_score += score

        print('Sum Score of Neural Networks Model : ', sum_score)
