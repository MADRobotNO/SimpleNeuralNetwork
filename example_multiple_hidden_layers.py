from NeuralNetwork import Model
from RandomData import HumanRandomData

nn = Model(size_of_input_layer=3, size_of_hidden_layer=3, size_of_output_layer=1, number_of_hidden_layers=2, debug=False)

training_data = HumanRandomData(200)
training_data.normalize_data(training_data.data)
nn.train_model(training_data.data, training_data.targets, number_of_epochs=500, learning_rate=0.2, data_shuffle=True, log=True)

test_data = HumanRandomData(10, normalized=True)
nn.test_model(test_data.data, test_data.targets, log=True)

real_data = HumanRandomData(10, normalized=True, debug=True)
nn.feedforward(real_data.data, log=True)

