import logging
from surround import Runner
from machine_learnt_sum_function_surround.stages import MachineLearntSumFunctionSurroundData
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)


class BatchRunner(Runner):
    def run(self, is_training=False):
        self.assembler.init_assembler(True)
        data = MachineLearntSumFunctionSurroundData()

        if is_training:
            # Load data to be processed
            data.input_data = pd.read_csv(os.path.abspath('data/numbers_train.csv'))
            print("Training Data Shape : ", data.input_data.shape)
            print('Training Input Data :', data.input_data)
        else:
            data.input_data = pd.read_csv(os.path.abspath('data/numbers_test.csv'))
            print("Testing Data Shape : ", data.input_data.shape)
            print('Testing Input Data :', data.input_data)

        # Run assembler
        self.assembler.run(data, is_training)

        logging.info("Batch Runner: %s", data.output_data)
