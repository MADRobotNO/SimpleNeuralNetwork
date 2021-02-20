import numpy as np
import random


class Node:

    bias = 0.0
    input_data = None
    input_sum_with_bias = 0.0
    output = 0.0
    weights = []
    error = 0
    size_of_input_layer = 0
    debug = False

    def __init__(self, size_of_input_layer, debug=False, weights=None, bias=None):
        self.debug = debug
        self.bias = round(self.generate_random_number(), 4)
        self.size_of_input_layer = size_of_input_layer

        if weights is not None:
            self.set_weights(weights)
        else:
            self.initialize_weights()
        if bias is not None:
            self.set_bias(bias)

    def set_input_data(self, input_data):
        self.input_data = list(input_data)

    def update_bias_with_error(self, error):
        self.bias = self.bias + error

    def set_error(self, error):
        self.error = error

    # manually sums all inputs multiplied by weights and add bias
    def calculate_input_sum(self):
        self.input_sum_with_bias = 0.0
        sum_string = ""
        for i in range(len(self.input_data)):
            sum_string += "{:.20f}".format((self.input_data[i] * self.weights[i])) + " (Input{0} ".format(i) \
                          + "{:.20f}".format(self.input_data[i]) + " * Weight{0} ".format(i) \
                          + "{:.20f}".format(self.weights[i]) + ") + "
            self.input_sum_with_bias += (self.input_data[i] * self.weights[i])
        sum_string += "Bias {:.20f}".format(self.bias)
        self.input_sum_with_bias += self.bias
        if self.debug:
            print("Calculating sum:", sum_string)
            print("Calculated sum:", self.input_sum_with_bias)

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def generate_output_data(self):
        self.calculate_input_sum()
        self.output = self.sigmoid(self.input_sum_with_bias)

        return self.output

    def initialize_weights(self):
        self.weights = []
        for i in range(self.size_of_input_layer):
            self.weights.append(round(self.generate_random_number(), 4))

    def set_weights(self, weights):
        self.weights = weights

    def set_bias(self, bias):
        self.bias = bias

    def update_weights_with_error(self, error):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + error[i]
        return self.weights

    def generate_random_number(self):
        return 2 * random.random() - 1

    def weights_to_print(self):
        string = ""
        for weight in self.weights:
            string += "{:.15f}".format(weight) + " "
        return string

    def bias_to_print(self):
        string = "{:.15f}".format(self.bias)
        return string
