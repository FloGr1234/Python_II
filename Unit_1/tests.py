import numpy as np
import os

if __name__ == "__main__":
    print(os.path.abspath(("TestOut1/0101.jpg")))
    print(os.path.exists("TestOut/001.jpg"))
    print("hallo world")
    A = np.array([[1,2,3],[5,6,7],[-4,3,10]])
    print(A)
    B = np.where(A<5,A*100,A-1)
    print(B)
    print(A)
    C = np.array([[1, 2, 3], [4, 4, 7], [-4, 3, 10]])

    D = np.where(A == C, True, False)
    print(D)