from ml import NLOneNeuron, NLTwoLayer, NLThreeLayer
from ml import case1
import numpy as np


def main():
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])
    training_outputs = np.array([[0, 1, 1, 0]]).T
    my_nl_one = NLOneNeuron(training_inputs=training_inputs, training_outputs=training_outputs)
    new_inputs = np.array([0, 0, 1])  # new situation
    my_nl_one.calculate(input_values=new_inputs)

    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 0, 1]])
    training_outputs = np.array([[0, 1, 1, 0]]).T

    my_nl_two = NLTwoLayer(training_inputs=training_inputs, training_outputs=training_outputs, hidden_neurons=4)
    my_nl_two.calculate(input_values=new_inputs)

    my_nl_three = NLThreeLayer(training_inputs=training_inputs, training_outputs=training_outputs, first_layer_neurons=4)
    my_nl_three.calculate(input_values=new_inputs)


if __name__ == '__main__':
    main()
