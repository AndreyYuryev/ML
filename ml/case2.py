from .stuff import sigmoid
import numpy as np


def case2():
    training_inputs = np.array([[0, 0, 1],
                                [0, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])
    training_outputs = np.array([[0, 1, 1, 0]]).T

    np.random.seed(1)

    synaptic_weights1 = 2 * np.random.random((3, 4)) - 1
    synaptic_weights2 = 2 * np.random.random((4, 1)) - 1

    print("Случайные стартовые веса")
    print("1", synaptic_weights1)
    print("2", synaptic_weights2)

    # Метод обратного распространения
    for i in range(60000):
        input_layer = training_inputs
        outputs1 = sigmoid(np.dot(input_layer, synaptic_weights1))
        outputs2 = sigmoid(np.dot(outputs1, synaptic_weights2))

        err2 = training_outputs - outputs2
        delta2 = err2 * (outputs2 * (1 - outputs2))
        err1 = np.dot(delta2, synaptic_weights2.T)
        delta1 = err1 * (outputs1 * (1 - outputs1))

        #                                    delta2
        # adjustments2 = np.dot(outputs1.T, err2 * (outputs2 * (1 - outputs2)))
        adjustments2 = np.dot(outputs1.T, delta2)
        synaptic_weights2 += adjustments2

        adjustments1 = np.dot(input_layer.T, delta1)
        synaptic_weights1 += adjustments1

    print("Веса после обучение")
    print(synaptic_weights1)
    print(synaptic_weights2)

    print("Результат после обучения")
    print(outputs1)
    print(outputs2)

    # тест

    new_inputs = np.array([1, 1, 0])  # new situation
    outputs1 = sigmoid(np.dot(new_inputs, synaptic_weights1))
    outputs2 = sigmoid(np.dot(outputs1, synaptic_weights2))
    print("Новая ситуация")
    print(outputs1)
    print(outputs2)
