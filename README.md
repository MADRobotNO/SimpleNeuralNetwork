# SimpleNeuralNetwork
A simple Neural Network library for learning porpouses  
  
Neural Networ creates an input layer, one hidden layer and output layer at this point.  
Library uses currently only sigmoid function as an activator.  
  
The constructor for Neural Network takes following parameters:  
size_of_input_layer - number of input nodes (or simply size of one data row)  
size_of_hidden_layer - number of hidden layer nodes  
size_of_output_layer - number of output layer nodes  
debug=False - boolean value to toggle debuging mode (extra informations are printed in the console)  
  
Possible functions:  
train_model(input_data, targets, numer_of_epochs=1, learning_rate=0.1) - this function train model  
test_model(input_data, targets) - this function is the same as feedforward but with check against target data  
feedforward(input_data) - this is function to use trained Neural Network  
  
Input data:  
It has to be an array of arrays. One row in main array contains an array of input values.  
Example:  
[[1,2,3],[4,5,6],[7,8,9],[1,5,9]]  
size_of_input_layer will be in this case 3 and input data set contains 4 sets.  
  
