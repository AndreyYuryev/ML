import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NLOneLayer:
    def __init__(self, training_inputs, training_outputs, input_neurons=3,  output_neurons=1):
        """ one layer with input neurons by default 3 and one output neurons
            output matrix should be transposed  .T
        """
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((input_neurons, output_neurons)) - 1

        print("Случайные стартовые веса")
        print(self.synaptic_weights)

        # Метод обратного распространения
        for i in range(60000):
            input_layer = training_inputs
            outputs = sigmoid(np.dot(input_layer, self.synaptic_weights))

            err = training_outputs - outputs
            adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))
            self.synaptic_weights += adjustments

        print("Веса после обучение")
        print(self.synaptic_weights)

    def calculate(self, input_values):
        outputs = sigmoid(np.dot(input_values, self.synaptic_weights))
        print("Новая ситуация")
        print(input_values, '-->', outputs)


class NLTwoLayer:
    def __init__(self, training_inputs, training_outputs, input_dim=3, fall=4, output_dim=1):
        """ two layer with input neurons by default 3 and one output neurons
            output matrix should be transposed  .T
        """
        np.random.seed(1)

        self.synaptic_weights1 = 2 * np.random.random((input_dim, fall)) - 1
        self.synaptic_weights2 = 2 * np.random.random((fall, output_dim)) - 1

        print("Случайные стартовые веса")
        print(self.synaptic_weights1, self.synaptic_weights2)

        # Метод обратного распространения
        for i in range(60000):
            input_layer = training_inputs
            outputs1 = sigmoid(np.dot(input_layer, self.synaptic_weights1))
            outputs2 = sigmoid(np.dot(outputs1, self.synaptic_weights2))

            err2 = training_outputs - outputs2
            delta2 = err2 * (outputs2 * (1 - outputs2))
            err1 = np.dot(delta2, self.synaptic_weights2.T)
            delta1 = err1 * (outputs1 * (1 - outputs1))

            #                                    delta2
            # adjustments2 = np.dot(outputs1.T, err2 * (outputs2 * (1 - outputs2)))
            adjustments2 = np.dot(outputs1.T, delta2)
            self.synaptic_weights2 += adjustments2

            adjustments1 = np.dot(input_layer.T, delta1)
            self.synaptic_weights1 += adjustments1

        print("Веса после обучение")
        print(self.synaptic_weights1, self.synaptic_weights2)

    def calculate(self, input_values):
        outputs1 = sigmoid(np.dot(input_values, self.synaptic_weights1))
        outputs2 = sigmoid(np.dot(outputs1, self.synaptic_weights2))
        print("Новая ситуация")
        print(input_values, '-->', outputs2)
