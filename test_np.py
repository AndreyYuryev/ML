from ml import NLOneLayer
import numpy as np


def main():
    training_inputs = np.loadtxt("ml/dataset_in.csv", delimiter=';', dtype=int)
    training_outputs = np.array([np.loadtxt("ml/dataset_out.csv", delimiter=';', dtype=int)]).T
    my_nl_one = NLOneLayer(training_inputs=training_inputs, training_outputs=training_outputs)
    new_inputs = np.array([0, 0, 1])  # new situation
    my_nl_one.calculate(input_values=new_inputs)


if __name__ == '__main__':
    main()
