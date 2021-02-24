# SimpleNeuralNetwork
A simple Neural Network library for learning purpose.  
  
Requires:
numpy

Not supported:
1. It is not possible to have different sizes for different hidden layers  
2. There is only one activation function - sigmoid  
3. It is not possible to have different activation functions on different layers  
4. Only one target output supported

Neural Network creates an input layer, hidden layers and output layer. 

The constructor for Neural Network takes following parameters:  
size_of_input_layer - number of input nodes (or simply size of one data row)  
size_of_hidden_layer - number of hidden layer perceptrons  
size_of_output_layer - number of output layer perceptrons  
number_of_hidden_layers=1 - number of hidden layers (default 1)  
debug=False - Boolean value to toggle debugging mode (extra information is printed in the console)  
  
Possible functions:  
train_model(input_data, targets, numer_of_epochs=1, learning_rate=0.1, data_shuffle=False, debug=False, log=False) - this function train model  
test_model(input_data, targets, debug=False, log=False) - this function is the same as feedforward but with check against target data
feedforward(input_data, debug=False, log=False) - this is function to use trained Neural Network  

Log parameter enables logging to CSV files that will be created in the project directory
debug=False - Boolean value to toggle debugging mode (extra information is printed in the console)
data_shuffle - enables shuffling of input data and targets. In some cases that might increase learning rate

Input data:  
It has to be an array of arrays. One row in main array contains an array of input values.  
Example:  
[[1,2,3],[4,5,6],[7,8,9],[1,5,9]]  
size_of_input_layer will be in this case 3 and input data set contains 4 sets.  
  
