from ml import NLOneNeuron, NLTwoLayer, NLThreeLayer
import numpy as np


def main():
    training_inputs = np.loadtxt("ml/dataset_in.csv", delimiter=';', dtype=int)
    training_outputs = np.array([np.loadtxt("ml/dataset_out.csv", delimiter=';', dtype=int)]).T
    my_nl_one = NLTwoLayer(training_inputs=training_inputs, training_outputs=training_outputs, input_numbers=4,
                           hidden_neurons=6)

    new_inputs = np.array([0, 0, 1, 0])  # new situation
    my_nl_one.calculate(input_values=new_inputs)
    new_inputs = np.array([1, 0, 1, 0])  # new situation
    my_nl_one.calculate(input_values=new_inputs)

    my_nl_three = NLThreeLayer(training_inputs=training_inputs, training_outputs=training_outputs,
                               input_dim=4, first_layer_neurons=4, second_layer_neurons=4)

    new_inputs = np.array([0, 0, 1, 0])  # new situation
    my_nl_three.calculate(input_values=new_inputs)
    new_inputs = np.array([1, 0, 1, 0])  # new situation
    my_nl_three.calculate(input_values=new_inputs)


if __name__ == '__main__':
    main()
