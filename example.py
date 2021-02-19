from NeuralNetwork import Model
from RandomData import HumanRandomData

nn = Model(size_of_input_layer=3, size_of_hidden_layer=7, size_of_output_layer=1)

training_data = HumanRandomData(200)
training_data.normalize_data(training_data.data)
nn.train_model(training_data.data, training_data.targets, number_of_epochs=2000, learning_rate=0.1)

test_data = HumanRandomData(20)
test_data.normalize_data(test_data.data)
nn.test_model(test_data.data, test_data.targets)

real_data = HumanRandomData(2, debug=True)
real_data.normalize_data(real_data.data)
nn.feedforward(real_data.data)
