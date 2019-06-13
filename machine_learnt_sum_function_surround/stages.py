from surround import Estimator, SurroundData, Validator
from machine_learnt_sum_function_surround.learn import LRLearn, NNLearn
from machine_learnt_sum_function_surround.validate import Validate

class MachineLearntSumFunctionSurroundData(SurroundData):
    input_data = None
    output_data = None
    test_data = None
    validate_data = None


class ValidateData(Validator):
    def validate(self, surround_data, config):
        print('Input Data Validation Passed')


class Main(Estimator):
    def estimate(self, surround_data, config):
        validator = Validate(surround_data.input_data)
        validator.__validate_regression_model__()
        validator.__validate_neural_networks_model__()

    def fit(self, surround_data, config):
        lr_learner = LRLearn(surround_data.input_data)
        lr_learner.__learn_linear_regression__()

        nn_learner = NNLearn(surround_data.input_data)
        nn_learner.__learn_neural_networks__()

        surround_data.output_data = surround_data.input_data
