from Perceptron import Perceptron
import copy
import random
import csv
import datetime


class Model:

    debug = False
    number_of_hidden_layers = 1
    input_data = []
    hidden_layers = []
    output_layer = []

    def __init__(self, size_of_input_layer, size_of_hidden_layer, size_of_output_layer, number_of_hidden_layers=1, debug=False):
        self.print_string_with_star_lines("### --- Initializing neural network --- ###")
        self.size_of_input_layer = size_of_input_layer
        self.size_of_hidden_layer = size_of_hidden_layer
        self.size_of_output_layer = size_of_output_layer
        self.number_of_hidden_layers = number_of_hidden_layers
        self.debug = debug
        self.initialize_hidden_layers()
        self.initialize_output_layer()
        self.print_string_with_star_lines("### --- Neural network layers initialized --- ###")

    def feed_data_to_input_layer(self, user_input):
        self.input_data = list(user_input)
        self.print_string_with_star_lines("Input layer initialized with {0} nodes".format(self.size_of_input_layer))
        self.print_details(self.input_data)
        return self.input_data

    def initialize_hidden_layers(self):
        for layer in range(self.number_of_hidden_layers):
            hidden_layer = []
            for i in range(self.size_of_hidden_layer):
                if layer == 0:
                    hidden_layer.append(Perceptron(self.size_of_input_layer, self.debug))
                else:
                    hidden_layer.append(Perceptron(self.size_of_hidden_layer, self.debug))
            self.print_string_with_star_lines("Hidden layer initialized with {0} nodes".format(len(hidden_layer)))
            self.print_details(hidden_layer)
            self.hidden_layers.append(hidden_layer)
        return self.hidden_layers

    def initialize_output_layer(self):
        for i in range(self.size_of_output_layer):
            self.output_layer.append(Perceptron(self.size_of_hidden_layer, self.debug))
        self.print_string_with_star_lines("Output layer initialized with {0} nodes".format(self.size_of_output_layer))
        self.print_details(self.output_layer)
        return self.output_layer

    def test_model(self, input_data, targets, debug=False, log=False):
        if debug:
            self.debug = True
        self.print_string_with_star_lines("### --- Test initialized --- ###")
        self.feed_data_to_input_layer(input_data)
        total_accuracy = 0

        filename = "log_test.csv"
        if log:
            now_date = datetime.datetime.now()
            now_date = now_date.strftime("%d-%m-%Y_%H-%M-%S")
            filename = "log_test_" + now_date + ".csv"
            with open(filename, mode='w', newline="") as log_file:
                data_writer = csv.writer(log_file, delimiter=';')
                row = ["Start test", now_date]
                data_writer.writerow(row)


        for input_row_index, data_set in enumerate(self.input_data):

            hidden_outputs = []
            outputs = []
            output_errors = []

            if self.debug:
                print("Data row:", input_row_index, "Data:", data_set)
                print("\n***Hidden layer***\n")

            for layer_index, layer in enumerate(self.hidden_layers):
                layer_hidden_outputs = []

                for node_index, node in enumerate(layer):

                    if layer_index == 0:
                        node.set_input_data(data_set)
                    else:
                        node.set_input_data(hidden_outputs[layer_index - 1])

                    if self.debug:
                        print("Node", node_index)
                        print("Node input data:", node.input_data)
                        print("Weights:", node.weights)
                        print("Bias:", node.bias)
                    output = node.generate_output_data()
                    layer_hidden_outputs.append(output)
                    if self.debug:
                        print("Output:", output)
                        print()

                hidden_outputs.append(layer_hidden_outputs)

                if self.debug:
                    print("Hidden outputs:", hidden_outputs)
                    print("\n***Output layer***\n")

            for node_index, node in enumerate(self.output_layer):
                node.set_input_data(hidden_outputs[self.number_of_hidden_layers - 1])
                if self.debug:
                    print("Node", node_index)
                    print("Node input data:", node.input_data)
                    print("Weights:", node.weights)
                    print("Bias:", node.bias)
                output = node.generate_output_data()
                outputs.append(output)
                error = targets[input_row_index] - output
                node.set_error(error)
                if self.debug:
                    print("Output:", output)
                    print("Target:", targets[input_row_index])
                    print("Output error:", error)
                    print()
                output_errors.append(error)

            if self.debug:
                print("Outputs for output layer:", outputs)
                print("All errors for data row", output_errors)
                print()

            if self.debug:
                print("Outputs for output layer:", outputs, "with correct answer", targets[input_row_index])
            total_error_for_output = 0
            for output_error in output_errors:
                total_error_for_output += abs(output_error)
            total_error_for_output = total_error_for_output / len(output_errors)
            total_accuracy_for_output = (1 - total_error_for_output)
            print(f"Accuracy for dataset #{input_row_index} is", round(total_accuracy_for_output * 100, 2),
                  "% and the error was", round(total_error_for_output * 100, 2), "%")
            if log:
                with open(filename, mode='a+', newline="") as log_file:
                    data_writer = csv.writer(log_file, delimiter=';')
                    row = [f"Accuracy for dataset is", round(total_accuracy_for_output * 100, 2),
                           "Error was", round(total_error_for_output * 100, 2)]
                    data_writer.writerow(row)
            total_accuracy += total_accuracy_for_output

        total_accuracy = total_accuracy/len(input_data)
        print("\nTotal accuracy for test:", round(total_accuracy*100, 2), "%\n")

        if log:
            now_date = datetime.datetime.now()
            now_date = now_date.strftime("%d-%m-%Y_%H-%M-%S")
            with open(filename, mode='a+', newline="") as log_file:
                data_writer = csv.writer(log_file, delimiter=';')
                row = ["End test", now_date]
                data_writer.writerow(row)
                row = ["Total accuracy for test:", round(total_accuracy*100, 2)]
                data_writer.writerow(row)

    def train_model(self, input_data, targets, number_of_epochs=1, learning_rate=0.1, data_shuffle=False, debug=False, log=False):
        if debug:
            self.debug = True
        self.feed_data_to_input_layer(input_data)

        self.best_result_model = []
        self.trained_model = []

        self.print_string_with_star_lines("### --- Training initialized --- ###")
        print("Number of epochs:", number_of_epochs)
        print("Learning rate:", learning_rate)
        self.print_string_with_star_lines()
        epoch_accuracy = 0
        max_accuracy = 0
        count_trainings = 0
        filename = "log_training.csv"
        if log:
            now_date = datetime.datetime.now()
            now_date = now_date.strftime("%d-%m-%Y_%H-%M-%S")
            filename = "log_training_" + now_date + ".csv"
            with open(filename, mode='w', newline="") as log_file:
                data_writer = csv.writer(log_file, delimiter=';')
                row = ["Start training", now_date]
                data_writer.writerow(row)
                row = ["Model settings:", "LR", learning_rate, "Numb.of.epochs", number_of_epochs,
                       "Numb.of hidden layers", self.number_of_hidden_layers, "Size of hidden layers",
                       self.size_of_hidden_layer, "Size of training data", len(self.input_data), "Data shuffle",
                       data_shuffle]
                data_writer.writerow(row)
        for epoch in range(1, number_of_epochs + 1):
            if self.debug:
                print("Epoch:", epoch)
            count_data_rows = 0
            epoch_error_rate = 0

            for input_row_index, data_set in enumerate(self.input_data):
                hidden_outputs = []
                outputs = []
                output_errors = []

                if self.debug:
                    print("Data row:", input_row_index)
                    print("\n***Hidden layer***\n")

                for layer_index, layer in enumerate(self.hidden_layers):
                    layer_hidden_outputs = []
                    for node_index, node in enumerate(layer):

                        if layer_index == 0:
                            node.set_input_data(data_set)
                        else:
                            node.set_input_data(hidden_outputs[layer_index-1])

                        if self.debug:
                            print("Node", node_index)
                            print("Node input data:", node.input_data)
                            print("Weights:", node.weights)
                            print("Bias:", node.bias)
                        output = node.generate_output_data()
                        layer_hidden_outputs.append(output)
                        if self.debug:
                            print("Output:", output)
                            print()

                    hidden_outputs.append(layer_hidden_outputs)

                    if self.debug:
                        print("Hidden outputs:", hidden_outputs)
                        print("\n***Output layer***\n")

                for node_index, node in enumerate(self.output_layer):
                    node.set_input_data(hidden_outputs[self.number_of_hidden_layers-1])
                    if self.debug:
                        print("Node", node_index)
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
                self.best_result_model = self.store_temp_model(self.hidden_layers, self.output_layer)
            print("Epoch {0} error rate:".format(epoch), "{:.16f}".format(epoch_error_rate),
                  "accuracy", "{:.2f}".format(epoch_accuracy), "%", " - Current max accuracy:", "{:.2f} %".format(max_accuracy))
            if log:
                with open(filename, mode='a+', newline="") as log_file:
                    data_writer = csv.writer(log_file, delimiter=';')
                    row = [epoch, "{:.2f}".format(epoch_accuracy), "{:.2f}".format(max_accuracy)]
                    data_writer.writerow(row)

            # shuffle input data
            if data_shuffle:
                total_data = list(zip(self.input_data, targets))
                random.shuffle(total_data)
                self.input_data, targets = zip(*total_data)

        print("Number of trainings:", count_trainings)
        print("Max accuracy:", "{:.2f} %".format(max_accuracy))

        self.trained_model = self.store_temp_model(self.hidden_layers, self.output_layer)

        now_date = datetime.datetime.now()
        now_date = now_date.strftime("%d-%m-%Y_%H-%M-%S")

        if log:
            with open(filename, mode='a+', newline="") as log_file:
                data_writer = csv.writer(log_file, delimiter=';')
                row = ["End training", now_date]
                data_writer.writerow(row)
                row = ["Number of trainings:", count_trainings]
                data_writer.writerow(row)
                row = ["Max accuracy:", "{:.2f} %".format(max_accuracy)]
                data_writer.writerow(row)

        user_input = input("If You want to save model at current stage press 1\nIf You want to save model at highest accuracy press 2\nType anythin to exit\n")
        if user_input == "1":
            filename = "trained_model_" + now_date + ".csv"
            with open(filename, mode='a+', newline="") as trained_model:
                data_writer = csv.writer(trained_model, delimiter=';')
                data_writer.writerows(self.trained_model)
                data_writer.writerow(["{:.2f} %".format(epoch_accuracy)])
        elif user_input == "2":
            filename = "best_model_" + now_date + ".csv"
            with open(filename, mode='a+', newline="") as best_model:
                data_writer = csv.writer(best_model, delimiter=';')
                data_writer.writerows(self.best_result_model)
                data_writer.writerow(["{:.2f} %".format(max_accuracy)])

    def store_temp_model(self, hidden_layers, output_layer):
        rows = []
        for hidden_layer in hidden_layers:
            for perceptron in hidden_layer:
                rows.append([perceptron.weights, perceptron.bias])
        for perceptron in output_layer:
            rows.append([perceptron.weights, perceptron.bias])
        return rows

    def backpropagation(self, inputs, hidden_outputs, outputs, output_errors, learning_rate):

        hidden_weights = self.hidden_weights()
        hidden_bias = self.hidden_bias()

        output_weights = self.output_weights()
        output_bias = self.output_bias()

        if self.debug:
            print("*** Backpropagation starts ***\n")
            print("Inputs:", inputs)
            print("Hidden outputs:", hidden_outputs)
            print("Hidden weights:", hidden_weights)
            print("Outputs", outputs)
            print("Output errors:", output_errors)

        hidden_errors = self.calculate_hidden_errors(output_errors, output_weights, hidden_weights, hidden_outputs)

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
            weight_update_values.append(self.calculate_weights_update_value(gradient, hidden_outputs[self.number_of_hidden_layers-1]))

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

        for layer_index in range(self.number_of_hidden_layers):
            last_layer = self.number_of_hidden_layers-1-layer_index
            if self.debug:
                print("Hidden layer", last_layer)

            weight_update_values = []
            bias_update_values = []

            for i, hidden_error in enumerate(hidden_errors[last_layer]):
                gradient = self.calculate_gradient(hidden_error, hidden_outputs[last_layer][i], learning_rate)
                bias_update_values.append(gradient)
                if last_layer == 0:
                    weight_update_values.append(self.calculate_weights_update_value(gradient, inputs))
                else:
                    weight_update_values.append(self.calculate_weights_update_value(gradient, hidden_outputs[last_layer]))

            if self.debug:
                print("Calculated update values for weights:", weight_update_values)
                print("Calculated update values for bias:", bias_update_values)

            for i, node in enumerate(self.hidden_layers[last_layer]):
                node.update_weights_with_error(weight_update_values[i])
                node.update_bias_with_error(bias_update_values[i])
                if self.debug:
                    print("Node {0}".format(i))
                    print("Update values for weights:", weight_update_values[i])
                    print("Old weights:", hidden_weights[last_layer][i])
                    print("Updated weights:", node.weights_to_print())
                    print("Old bias:", hidden_bias[last_layer][i])
                    print("Updated bias:", "{:.20f}".format(node.bias))
                    print()

    def calculate_hidden_errors(self, output_errors, output_weights, hidden_weights, hidden_outputs):
        if self.debug:
            print("Calculating hidden errors:")
            print("Output errors:", output_errors, "output weights", output_weights)
        hidden_errors = []
        for layer_index in range(self.number_of_hidden_layers):
            hidden_layer_errors = []
            for i, node in enumerate(self.hidden_layers[self.number_of_hidden_layers-layer_index-1]):
                error = 0
                if layer_index == 0:
                    for j, output_error in enumerate(output_errors):
                        error += output_error * output_weights[j][i]
                else:
                    for j, hidden_error in enumerate(hidden_errors[layer_index-1]):
                        error += hidden_error * hidden_weights[self.number_of_hidden_layers-layer_index][j][i]
                node.set_error(error)
                hidden_layer_errors.append(error)
            hidden_errors.append(hidden_layer_errors)
            if self.debug:
                print("Hidden errors for hidden layer", self.number_of_hidden_layers-layer_index-1, ":", hidden_errors[layer_index])

        # rearrange array so that it is arranged same way
        hidden_errors_rearanged = []
        for layer_index in range(len(hidden_errors)):
            hidden_errors_rearanged.append(hidden_errors[len(hidden_errors)-layer_index-1])

        return hidden_errors_rearanged

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
        for layer in self.hidden_layers:
            hidden_layer_weights = []
            for i, node in enumerate(layer):
                hidden_layer_weights.append(copy.copy(layer[i].weights))
            hidden_weights.append(hidden_layer_weights)
        return hidden_weights

    def output_weights(self):
        output_weights = []
        for i, node in enumerate(self.output_layer):
            output_weights.append(copy.copy(self.output_layer[i].weights))
        return output_weights

    def hidden_bias(self):
        hidden_bias = []
        for layer in self.hidden_layers:
            hidden_layer_bias = []
            for i, node in enumerate(layer):
                hidden_layer_bias.append(copy.copy(layer[i].bias))
            hidden_bias.append(hidden_layer_bias)
        return hidden_bias

    def output_bias(self):
        output_bias = []
        for i, node in enumerate(self.output_layer):
            output_bias.append(copy.copy(self.output_layer[i].bias))
        return output_bias

    def feedforward(self, input_data, debug=False, log=False):

        if debug:
            self.debug = True
        self.print_string_with_star_lines("### --- Feed forward initialized --- ###")
        self.feed_data_to_input_layer(input_data)

        filename = "log_feedforward.csv"
        if log:
            now_date = datetime.datetime.now()
            now_date = now_date.strftime("%d-%m-%Y_%H-%M-%S")
            filename = "log_feedforward_" + now_date + ".csv"
            with open(filename, mode='w', newline="") as log_file:
                data_writer = csv.writer(log_file, delimiter=';')
                row = ["Start feedforward", now_date]
                data_writer.writerow(row)

        for input_row_index, data_set in enumerate(self.input_data):

            hidden_outputs = []
            outputs = []

            if self.debug:
                print("Data row:", input_row_index, "Data:", data_set)
                print("\n***Hidden layer***\n")

            for layer_index, layer in enumerate(self.hidden_layers):
                layer_hidden_outputs = []

                for node_index, node in enumerate(layer):

                    if layer_index == 0:
                        node.set_input_data(data_set)
                    else:
                        node.set_input_data(hidden_outputs[layer_index - 1])

                    if self.debug:
                        print("Node", node_index)
                        print("Node input data:", node.input_data)
                        print("Weights:", node.weights)
                        print("Bias:", node.bias)
                    output = node.generate_output_data()
                    layer_hidden_outputs.append(output)
                    if self.debug:
                        print("Output:", output)
                        print()

                hidden_outputs.append(layer_hidden_outputs)

                if self.debug:
                    print("Hidden outputs:", hidden_outputs)
                    print("\n***Output layer***\n")

            for node_index, node in enumerate(self.output_layer):
                node.set_input_data(hidden_outputs[self.number_of_hidden_layers - 1])
                if self.debug:
                    print("Node", node_index)
                    print("Node input data:", node.input_data)
                    print("Weights:", node.weights)
                    print("Bias:", node.bias)
                output = node.generate_output_data()
                outputs.append(output)
                if self.debug:
                    print("Output:", output)
                    print()

            print("Outputs", outputs, "for dataset", data_set)
            if log:
                with open(filename, mode='a+', newline="") as log_file:
                    data_writer = csv.writer(log_file, delimiter=';')
                    row = [data_set, outputs]
                    data_writer.writerow(row)

        print()
        if log:
            now_date = datetime.datetime.now()
            now_date = now_date.strftime("%d-%m-%Y_%H-%M-%S")
            with open(filename, mode='a+', newline="") as log_file:
                data_writer = csv.writer(log_file, delimiter=';')
                row = ["End feedforward", now_date]
                data_writer.writerow(row)

    def print_details(self, data):
        if self.debug:
            count = 0
            for element in data:
                if isinstance(element, Perceptron):
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

    def load_model_from_file(self, file_path):
        with open(file_path) as model_file:
            data_reader = csv.reader(model_file, delimiter=';')

            hidden_done = False
            hidden_layers_index = 0
            row_index = 0
            for row in data_reader:
                new_list = eval(row[0])
                if hidden_done:
                    for perceptron in self.output_layer:
                        perceptron.set_weights(new_list)
                        perceptron.set_bias(float(row[1]))

                else:
                    new_list = eval(row[0])
                    self.hidden_layers[hidden_layers_index][row_index].set_weights(new_list)
                    self.hidden_layers[hidden_layers_index][row_index].set_bias(float(row[1]))
                    if row_index == self.size_of_hidden_layer - 1:
                        hidden_layers_index += 1
                    if hidden_layers_index == self.number_of_hidden_layers:
                        hidden_done = True
                row_index += 1
        print("Model loaded successfully")
