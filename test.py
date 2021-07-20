from ml import NLOneLayer, NLTwoLayer
from ml import case1
import numpy as np


def main():
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])
    training_outputs = np.array([[0],
                                 [1],
                                 [1],
                                 [0]]).T
    my_nl = NLOneLayer(training_inputs=training_inputs, training_outputs=training_outputs)
    new_inputs = np.array([0, 0, 1])  # new situation
    my_nl.calculate(input_values=new_inputs)


if __name__ == '__main__':
    main()
