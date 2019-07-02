# libraries
import h5py
import yaml as yaml
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


def predict_linear_regression():
    with open("configs_predict.yaml", 'r') as stream:
        try:
            configs = yaml.safe_load(stream)
            print(configs)
        except yaml.YAMLError as exc:
            print(exc)

    a = configs.get('a')
    b = configs.get('b')
    c = configs.get('c')
    d = configs.get('d')

    print('Predict Linear Regression Model')
    h5f = h5py.File(os.path.abspath('../models/linear_reg_model.hdf5'), 'r')
    intercept = h5f['intercept'][()]
    coefficients = h5f['coefficients'][()]
    h5f.close()

    y_actual = a + b + c + d
    print('Actual Sum Function of {} and {} and {} and {} is : {}'.format(a, b, c, d, y_actual))

    y_model = coefficients[0] * a + coefficients[1] * b + coefficients[2] * c + coefficients[3] * d + intercept
    print('Machine Learnt Sum Function of {} and {} and {} and {} is : {}'.format(a, b, c, d, y_model))


def predict_neural_networks():
    with open("configs_predict.yaml", 'r') as stream:
        try:
            configs = yaml.safe_load(stream)
            print(configs)
        except yaml.YAMLError as exc:
            print(exc)

    a = configs.get('a')
    b = configs.get('b')
    c = configs.get('c')
    d = configs.get('d')
    x = pd.DataFrame({'a': [a], 'b': [b], 'c': [c], 'd': [d]})
    print('Predict X array : ', x)
    y = a + b + c + d

    print('Predict Neural Networks Model')

    # model = load_model(os.path.abspath('../models/neural_networks_model.h5'))
    # model.load_weights(os.path.abspath('../models/nn_model_weights.h5'))
    model = Sequential()
    model.add(Dense(1, kernel_initializer='uniform', activation='relu', input_dim=4))
    sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    score, accuracy = model.evaluate(x, y)

    print('Predict Neural Networks Model Score', score)
    print('Predict Neural Networks Model Accuracy', accuracy)


if __name__ == "__main__":
    predict_linear_regression()
    predict_neural_networks()