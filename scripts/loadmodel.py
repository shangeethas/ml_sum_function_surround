import h5py
import os


def __validate_regression_model():
    h5f1 = h5py.File(os.path.abspath('linear_reg_model_5.hdf5'), 'w')
    h5f1.create_dataset("default", (100,))
    h5f1.create_dataset("ints", (100,), dtype='i8')
    h5f1.close()

    # print('Validate Regression Model : ', os.path.abspath('models/linear_reg_model.hdf5'))
    h5f = h5py.File(os.path.abspath('linear_reg_model_5.hdf5'), 'r')
    default = h5f.get('default')
    ints = h5f.get('ints')
    h5f.close()

    print('Intercept : ', default)
    print('Coefficients: ', ints)


if __name__ == "__main__":
    __validate_regression_model()