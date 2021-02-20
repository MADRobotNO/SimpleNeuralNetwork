from Node import Node
import copy
import random


class Model:

    debug = False
    input_data = []
    hidden_layer = []
    output_layer = []

    def __init__(self, size_of_input_layer, size_of_hidden_layer, size_of_output_layer, debug=False):
        self.print_string_with_star_lines("### --- Initializing neural network --- ###")
        self.size_of_input_layer = size_of_input_layer
        self.size_of_hidden_layer = size_of_hidden_layer
        self.size_of_output_layer = size_of_output_layer
        self.debug = debug
        self.initialize_hidden_layer()
        self.initialize_output_layer()
        self.print_string_with_star_lines("### --- Neural network layers initialized --- ###")

    def feed_data_to_input_layer(self, user_input):
        self.input_data = list(user_input)
        self.print_string_with_star_lines("Input layer initialized with {0} nodes".format(self.size_of_input_layer))
        self.print_details(self.input_data)
        return self.input_data

    def initialize_hidden_layer(self):
        for i in range(self.size_of_hidden_layer):
            self.hidden_layer.append(Node(self.size_of_input_layer, self.debug))
        self.print_string_with_star_lines("Hidden layer initialized with {0} nodes".format(self.size_of_hidden_layer))
        self.print_details(self.hidden_layer)
        return self.hidden_layer

    def initialize_output_layer(self):
        for i in range(self.size_of_output_layer):
            self.output_layer.append(Node(self.size_of_hidden_layer, self.debug))
        self.print_string_with_star_lines("Output layer initialized with {0} nodes".format(self.size_of_output_layer))
        self.print_details(self.output_layer)
        return self.output_layer

    def test_model(self, input_data, targets, debug=False):
        if debug:
            self.debug = True
        self.print_string_with_star_lines("### --- Test initialized --- ###")
        self.feed_data_to_input_layer(input_data)

        total_accuracy = 0

        for index_data_set, data_set in enumerate(self.input_data):

            if self.debug:
                print("Data row", data_set)
                print("\n***Hidden layer***\n")

            hidden_outputs = []
            outputs = []
            count_nodes = 0
            output_errors = []

            for node in self.hidden_layer:
                node.set_input_data(data_set)
                output = node.generate_output_data()
                hidden_outputs.append(output)

                if self.debug:
                    print("Node", count_nodes)
                    print("Node input data:", node.input_data)
                    print("Weights:", node.weights)
                    print("Bias:", node.bias)
                    print("Output:", output)
                    print()

                count_nodes += 1

            if self.debug:
                print("\n***Output layer***\n")

            count_nodes = 0  # reset counter

            for i, node in enumerate(self.output_layer):
                node.set_input_data(hidden_outputs)
                output = node.generate_output_data()
                outputs.append(output)
                error = targets[index_data_set] - output
                node.set_error(error)
                if self.debug:
                    print("Node", count_nodes)
                    print("Node input data:", node.input_data)
                    print("Weights:", node.weights)
                    print("Bias:", node.bias)
                    print("Output:", output)
                    print("Target:", targets[index_data_set])
                    print("Output error:", error)
                    print()
                count_nodes += 1
                output_errors.append(error)

            if self.debug:
                print("Outputs for output layer:", outputs, "with correct answer", targets[index_data_set])
            total_error_for_output = 0
            for output_error in output_errors:
                total_error_for_output += abs(output_error)
            total_error_for_output = total_error_for_output/len(output_errors)
            total_accuracy_for_output = (1-total_error_for_output)
            print(f"Accuracy for dataset #{index_data_set} is", round(total_accuracy_for_output * 100, 2),
                  "% and the error was", round(total_error_for_output * 100, 2), "%")
            total_accuracy += total_accuracy_for_output

        total_accuracy = total_accuracy/len(input_data)
        print("\nTotal accuracy for test:", round(total_accuracy*100, 2), "%\n")

    def train_model(self, input_data, targets, number_of_epochs=1, learning_rate=0.1, data_shuffle=False):
        self.feed_data_to_input_layer(input_data)

        self.print_string_with_star_lines("### --- Training initialized --- ###")
        print("Number of epochs:", number_of_epochs)
        print("Learning rate:", learning_rate)
        self.print_string_with_star_lines()

        max_accuracy = 0
        count_trainings = 0

        for epoch in range(1, number_of_epochs + 1):
            if self.debug:
                print("Epoch:", epoch)
            count_data_rows = 0
            epoch_error_rate = 0

            for input_row_index, data_set in enumerate(self.input_data):
                count_nodes = 0  # only for debugging and printing, not used in calculations
                hidden_outputs = []
                outputs = []
                output_errors = []

                if self.debug:
                    print("Data row:", count_data_rows)
                    print("\n***Hidden layer***\n")

                for node in self.hidden_layer:
                    node.set_input_data(data_set)
                    if self.debug:
                        print("Node", count_nodes)
                        print("Node input data:", node.input_data)
                        print("Weights:", node.weights)
                        print("Bias:", node.bias)
                    output = node.generate_output_data()
                    hidden_outputs.append(output)
                    if self.debug:
                        print("Output:", output)
                        print()

                    count_nodes += 1

                if self.debug:
                    print("Hidden outputs:", hidden_outputs)
                    print("\n***Output layer***\n")

                count_nodes = 0  # reset counter

                for i, node in enumerate(self.output_layer):
                    node.set_input_data(hidden_outputs)
                    if self.debug:
                        print("Node", count_nodes)
                        print("Node input data:", node.input_data)
                        print("Weights:", node.weights)
                        print("Bias:", node.bias)
                    output = node.generate_output_data()
                    outputs.append(output)
                    error = targets[input_row_index]-output
                    node.set_error(error)
                    if self.debug:
                        print("Output:", output)
                        print("Target:", targets[input_row_index])
                        print("Output error:", error)
                        print()
                    count_nodes += 1
                    output_errors.append(error)

                if self.debug:
                    print("Outputs for output layer:", outputs)
                    print("All errors for data row", output_errors)
                    print()

                count_data_rows += 1
                count_trainings += 1
                row_error = 0

                for element in output_errors:
                    row_error += abs(element)
                row_error = row_error/len(output_errors)
                epoch_error_rate += row_error
                self.backpropagation(data_set, hidden_outputs, outputs, output_errors, learning_rate)

            epoch_error_rate = epoch_error_rate / count_data_rows
            epoch_accuracy = (1-epoch_error_rate)*100
            if epoch_accuracy > max_accuracy:
                max_accuracy = epoch_accuracy
            print("Epoch {0} error rate:".format(epoch), "{:.16f}".format(epoch_error_rate),
                  "accuracy", "{:.2f}".format(epoch_accuracy), "%")

            # shuffle input data
            if data_shuffle:
                total_data = list(zip(self.input_data, targets))
                random.shuffle(total_data)
                self.input_data, targets = zip(*total_data)

        print("Number of trainings:", count_trainings)
        print("Max accuracy:", "{:.2f}".format(max_accuracy), "%")

    def backpropagation(self, inputs, hidden_outputs, outputs, output_errors, learning_rate):

        hidden_weights = self.hidden_weights()
        hidden_bias = self.hidden_bias()

        output_weights = self.output_weights()
        output_bias = self.output_bias()

        if self.debug:
            print("*** Backpropagation starts ***\n")
            print("Inputs:", inputs)
            print("Hidden outputs:", hidden_outputs)
            print("Outputs", outputs)
            print("Output errors:", output_errors)

        hidden_errors = self.calculate_hidden_error(output_errors, output_weights)

        if self.debug:
            print("Hidden errors:", hidden_errors)
            print("Learning rate:", learning_rate)
            print()
            print("Output layer\n")

        weight_update_values = []
        bias_update_values = []

        for i, output_error in enumerate(output_errors):
            gradient = self.calculate_gradient(output_error, outputs[i], learning_rate)
            bias_update_values.append(gradient)
            weight_update_values.append(self.calculate_weights_update_value(gradient, hidden_outputs))

        if self.debug:
            print("Calculated update values for weights:", weight_update_values)
            print("Calculated update values for bias:", bias_update_values)

        for i, node in enumerate(self.output_layer):
            node.update_weights_with_error(weight_update_values[i])
            node.update_bias_with_error(bias_update_values[i])
            if self.debug:
                print("Node {0}".format(i))
                print("Update values for weights:", weight_update_values[i])
                print("Old weights:", output_weights[i])
                print("Updated weights:", node.weights_to_print())
                print("Old bias:", output_bias[i])
                print("Updated bias:", "{:.20f}".format(node.bias))
                print()

        if self.debug:
            print("Hidden layer\n")

        weight_update_values = []
        bias_update_values = []

        for i, hidden_error in enumerate(hidden_errors):
            gradient = self.calculate_gradient(hidden_error, hidden_outputs[i], learning_rate)
            bias_update_values.append(gradient)
            weight_update_values.append(self.calculate_weights_update_value(gradient, inputs))

        if self.debug:
            print("Calculated update values for weights:", weight_update_values)
            print("Calculated update values for bias:", bias_update_values)

        for i, node in enumerate(self.hidden_layer):
            node.update_weights_with_error(weight_update_values[i])
            node.update_bias_with_error(bias_update_values[i])
            if self.debug:
                print("Node {0}".format(i))
                print("Update values for weights:", weight_update_values[i])
                print("Old weights:", hidden_weights[i])
                print("Updated weights:", node.weights_to_print())
                print("Old bias:", hidden_bias[i])
                print("Updated bias:", "{:.20f}".format(node.bias))
                print()

    def calculate_hidden_error(self, output_errors, output_weights):
        if self.debug:
            print("Calculating hidden errors:")
        hidden_errors = []
        for i, node in enumerate(self.hidden_layer):
            error = 0
            for j, output_error in enumerate(output_errors):
                error += output_error * output_weights[j][i]
            node.set_error(error)
            hidden_errors.append(error)
        return hidden_errors

    def calculate_gradient(self, output_error, output, learning_rate):
        """
        Calculating gradient
        Formula:
        gradient = sigmoid_derivative(output_from_output_node) * output_error_from_output_node * learning_rate
        """

        gradient = self.sigmoid_derivative(output)
        gradient = gradient * output_error
        gradient = gradient * learning_rate
        if self.debug:
            print("Gradient", gradient)

        return gradient

    def calculate_weights_update_value(self, gradient, outputs_from_previous_layer):
        """
        Calculating delta for weight update
        Formula:
        Delta_weight{index} = gradient * input_data_from_previous_layer
        """

        update_values = []
        for output_from_previous_layer in outputs_from_previous_layer:
            value = gradient * output_from_previous_layer
            update_values.append(value)

        return update_values

    def sigmoid_derivative(self, value):
        """
        value has to be output of sigmoid function
        """
        return value * (1 - value)

    def hidden_weights(self):
        hidden_weights = []
        for i, node in enumerate(self.hidden_layer):
            hidden_weights.append(copy.copy(self.hidden_layer[i].weights))
        return hidden_weights

    def output_weights(self):
        output_weights = []
        for i, node in enumerate(self.output_layer):
            output_weights.append(copy.copy(self.output_layer[i].weights))
        return output_weights

    def hidden_bias(self):
        hidden_bias = []
        for i, node in enumerate(self.hidden_layer):
            hidden_bias.append(copy.copy(self.hidden_layer[i].bias))
        return hidden_bias

    def output_bias(self):
        output_bias = []
        for i, node in enumerate(self.output_layer):
            output_bias.append(copy.copy(self.output_layer[i].bias))
        return output_bias

    def feedforward(self, input_data, debug=False):
        self.print_string_with_star_lines("### --- Feed forward initialized --- ###")

        self.feed_data_to_input_layer(input_data)

        if debug:
            self.debug = True
        for data_set in self.input_data:

            if self.debug:
                print("Data row", data_set)
                print("\n***Hidden layer***\n")

            hidden_outputs = []
            outputs = []
            count_nodes = 0

            for node in self.hidden_layer:
                node.set_input_data(data_set)
                output = node.generate_output_data()
                hidden_outputs.append(output)

                if self.debug:
                    print("Node", count_nodes)
                    print("Node input data:", node.input_data)
                    print("Weights:", node.weights)
                    print("Bias:", node.bias)
                    print("Output:", output)
                    print()

                count_nodes += 1

            if self.debug:
                print("\n***Output layer***\n")

            count_nodes = 0  # reset counter

            for i, node in enumerate(self.output_layer):
                node.set_input_data(hidden_outputs)
                output = node.generate_output_data()
                outputs.append(output)
                if self.debug:
                    print("Node", count_nodes)
                    print("Node input data:", node.input_data)
                    print("Weights:", node.weights)
                    print("Bias:", node.bias)
                    print("Output:", output)
                    print()
                count_nodes += 1

            print("Data row", data_set)
            print("Outputs for output layer:", outputs)
            print()

    def randomize_factor(self):
        return random.random()

    def print_details(self, data):
        if self.debug:
            count = 0
            for element in data:
                if isinstance(element, Node):
                    print("Node:", count)
                    print("Weights:", element.weights)
                    print("Bias:", element.bias)
                    print()
                    count += 1
                else:
                    print(element)
                    count += 1
                    if count == len(data):
                        print()

    def print_string_with_star_lines(self, text=None):
        if text is None:
            print("\n**********************************************\n")
        else:
            print("*"*len(text))
            print(text)
            print("*"*len(text))
            print()
