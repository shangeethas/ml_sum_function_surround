# Machine Learned Sum Function

Machine Learning (ML) increasingly governs our modern society and it is vital that errors are defined and anomalies are detected for any given scenario. 
The sum function is considered to unveil the errors and the anomalies, as sum is the elemental function from a mathematician's perspective.
In simple terms, the sum of an addend 5 with another addend 7 is 12. In this exercise, the sum function of four inputs is taken into consideration.
Two learning strategies, Linear Regression (LR) and Neural Networks(NN) are designed to learn the sum function.

![alt text](common/logo.png "Logo")

## Prerequisites
* Python 3+ (Implemented on 3.7.3)
* surround 0.0.9
* numpy 1.16.3
* pandas 0.24.2
* PyYaml 5.1
* h5py 2.9.0
* keras 2.2.4
* tensorflow 1.13.1

## Experiments and Results
The entire experiment was carried out on MacOS with 2.3 GHz Intel Core i5 processor and 8GB memory. 
Four addends are referred as a, b, c and d and output is referred to as y. This exercise consists of four phases such as data generation phase, learning phase, validation phase and prediction phase.

### Data Generation Phase

A python utility script generates data points required for sum function. Configurations need to be explicitly specified in configurations file of yaml format.
#### Structure of configs_data.yaml
* random seed of input a
* random seed of input b
* random seed of input c
* random seed of input d
* required number of data points
* minimum range of data points
* maximum range of data points

A sample of configs_data.yaml is shown below
```
random_seed_a: 12
random_seed_b: 34
random_seed_c: 56
random_seed_d: 78
number_points: 1000000
min_range: 1
max_range: 100
```
From the generated data set, three partitions are created such as training (60%), testing (20%) and validation (20%) and are saved as "numbers_train.csv", "numbers_test.csv" and "numbers_validate.csv" respectively.
Further, correlation matrices are calculated for each partition. 

### Learning Phase
In general, hyper parameters for models are specified using following three strategies.
* initialisation with values offered by ML frameworks
* manual configurations based on recommendations from literature or experience
* trial and error

Following surround inbuilt command is used for training model based on training data set. 
```
python3 -m machine_learned_sum_function --mode train
```
#### LR model 
sci-kit learn python library is used to learn sum function and to find four regression coefficients and intercept.

|Hyper parameter                |Parameter value         | 
|:-----------------------------:|:---------------------: |
|intercept                      | -3.979039320256561e-13 |
|coefficient of a               |   1.                   |
|coefficient of b               |   1.                   |
|coefficient of c               |   1.                   |
|coefficient of d               |   1.                   |

#### NN model
Keras python library is used to construct NN model and to find weights and bias.
Following defined optimizers are used. SGD stands for Stochastic Gradient Descent.

|Optimizer Number | Optimizer Type |  Learning Rate |Clipping Normal Maximum Value|
|:---------------:|:--------------:|:--------------:|:---------------------------:|
|1                | SGD            |0.01            |1.                           |
|2                | SGD            |0.001           |1.                           |

Following defined NN models are used.

|Model Number |Model     |No of Layers  |Layer Description           |Kernel Initializer|Activation|Optimizer Number |
|:----------: |:-----:   |:-----------: |:-----------------:         |:----------------:|:--------:|:---------------:|
|1            |Sequential| 1            |Regular densely-connected   |uniform           |  relu    | 1               | 
|2            |Sequential| 1            |Regular densely-connected   |uniform           |  relu    | 2               |
|3            |Sequential| 1            |Regular densely-connected   |random uniform    |  relu    | 1               |


After completion of learning, both models are saved in h5 format. An epoch is a single pass through the entire training set, during iterative training of a neural network, followed by testing of the verification set.
Model 1 and Model 3 are trained using 12 epochs whereas Model 2 is trained only with 4 epochs, due its lower learning rate.


### Validation Phase
Following surround inbuilt command is used for validating model based on validation data set.
```
python3 -m machine_learned_sum_function --mode batch
```
The entire validation for all designed models took approximately 15 seconds. Following illustrates the validation metrics of different learned models.

|Model Number | Learning Strategy| Validation Metric          | Metric Value        |
|:-----------:|:----------------:|:-----------------:         |:-------------------:|
|1            | LR               | Mean sum of squared errors |6.334752934145276e-26|
|2            | NN               | score                      |0.2695742835038848   |
|2            | NN               | accuracy                   |0.5101575507858168   |
|3            | NN               | score                      |0.0018413071703050267|
|3            | NN               | accuracy                   |1.0                  |
|4            | NN               | score                      |0.3752033055084647   |
|4            | NN               | accuracy                   |0.31039655198171684  |

### Prediction Phase
As Surround library does not have prediction as one of its stages, a separate utility script was written to predict the sum of four unsigned floats, under four broader spaces, within trained ranges, beyond trained ranges, at border values and for all zeros.

LR Model outcomes as follows

|Addend a | Addend b | Addend c | Addend d | Ground truth outcome | Predicted model outcome |
|:-------:|:--------:|:--------:|:--------:|:--------------------:|:-----------------------:|
|12.00    | 24.00    | 36.00    | 48.00    | 120.00               |  119.99999999999973     |
|7.12     | 14.24    | 21.36    | 28.48    | 71.20                |   71.19999999999968     |
|-100.19  | -200.28  | -300.37  | -400.46  | -1001.30             | -1001.3000000000015     |
|100.19   |200.28    |300.37    |400.46    |1001.30               |  1001.3000000000006     |
|1.00     |1.00      |1.00      |1.00      |4.00                  |  3.99999999999961       |
|100.00   |100.00    |100.00    |100.00    |400.00                |   3.9909189             |
|0.00     |0.00      |0.00      |0.00      |0.00                  | -3.979039320256561e-13  |

NN Model 1 outcomes as follows

|Addend a | Addend b | Addend c | Addend d | Ground truth outcome | Predicted model outcome |
|:-------:|:--------:|:--------:|:--------:|:--------------------:|:-----------------------:|
|12.00    | 24.00    | 36.00    | 48.00    | 120.00               |     119.65047           |
|7.12     | 14.24    | 21.36    | 28.48    | 71.20                |      70.99509           |
|-100.19  | -200.28  | -300.37  | -400.46  | -1001.30             |          0.             |
|100.19   |200.28    |300.37    |400.46    |1001.30               |     998.33875           |
|1.00     |1.00      |1.00      |1.00      |4.00                  |    3.9951391            |
|100.00   |100.00    |100.00    |100.00    |400.00                |    398.9104             |
|0.00     |0.00      |0.00      |0.00      |0.00                  |   0.00609638            |

NN Model 2 outcomes as follows

|Addend a | Addend b | Addend c | Addend d | Ground truth outcome | Predicted model outcome |
|:-------:|:--------:|:--------:|:--------:|:--------------------:|:-----------------------:|
|12.00    | 24.00    | 36.00    | 48.00    | 120.00               |        119.97312        |
|7.12     | 14.24    | 21.36    | 28.48    | 71.20                |    71.186745            |
|-100.19  | -200.28  | -300.37  | -400.46  | -1001.30             |        0.               |
|100.19   |200.28    |300.37    |400.46    |1001.30               |    1001.0271            |
|1.00     |1.00      |1.00      |1.00      |4.00                  |    4.0055857            |
|100.00   |100.00    |100.00    |100.00    |400.00                |     399.90277           |
|0.00     |0.00      |0.00      |0.00      |0.00                  |     0.00662439          |

NN Model 3 outcomes as follows

|Addend a | Addend b | Addend c | Addend d | Ground truth outcome | Predicted model outcome |
|:-------:|:--------:|:--------:|:--------:|:--------------------:|:-----------------------:|
|12.00    | 24.00    | 36.00    | 48.00    | 120.00               |    119.69286            |
|7.12     | 14.24    | 21.36    | 28.48    | 71.2                 |    71.02048             |
|-100.19  | -200.28  | -300.37  | -400.46  | -1001.3              |     0.                  |
|100.19   |200.28    |300.37    |400.46    |1001.3                |    998.6883             |
|1.00     |1.00      |1.00      |1.00      |4.00                  |    3.9963598            |
|100.00   |100.00    |100.00    |100.00    |400.00                |    398.97647            |
|0.00     |0.00      |0.00      |0.00      |0.00                  |   0.00666155            |