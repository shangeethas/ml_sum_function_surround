# Machine Learned Sum Function

Machine Learning (ML) increasingly governs our modern society and it is vital that errors are defined and anomalies are detected for any given scenario. 
Sum function is considered to unveil the errors and the anomalies, as sum is the elemental function from a mathematician's perspective.

* Sum function is considered for four inputs and for one output.
* Two learning models, Linear Regression and Neural Networks are designed to learn the sum function.

![alt text](common/logo.png "Logo")

## Installation

### Prerequisites
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

### Data Generation Phase
Four inputs are referred as a, b, c and d and output is referred to as y.
gendata.py generates data points required for sum function. Configurations need to be explicitly specified in configs_data.yaml file.
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
Following is the surround inbuilt command for training model based on training data set.
`python3 -m machine_learned_sum_function --mode train`
#### Linear Regression model 
sci-kit learn python library is used to learn sum function and to find four regression coefficients and intercept.

#### Neural Networks model
Keras python library is used to construct NN model and to find weights and bias.

|Architecture Number |Model     |No of Layers  |No of Input Neurons|No of Output Neurons|
|:-----------------:|:-----:    |:-----------: |:-----------------:|:------------------:|  
|1                   |Sequential| 1            | 1                 |   1                |
 

After completion of learning, both models are saved in h5 format.

### Validation Phase
Following is the surround inbuilt command for validating model based on validation data set.
`python3 -m machine_learned_sum_function --mode batch`