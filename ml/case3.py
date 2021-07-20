import numpy as np


def case3():
   print("test")
   # sets = np.array([[0, 1, 1, 0]])
   sets = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])
   # np.savetxt("ml/dataset_in.csv", X=sets, delimiter=';')
   # data = np.loadtxt(fname="ml/test.txt")
   # data = np.loadtxt("ml/dataset_in.csv", delimiter=';', dtype=float)
   data = np.loadtxt("ml/dataset_in.csv", delimiter=';')
   print(data)
   print(sets)