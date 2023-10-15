import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


def householder(matrix):  # This procedure is presented in Numerical Analysis by Burden and Faires.
    a = np.copy(matrix)
    n = np.shape(a)[0]  # Get size of square matrix
    dim = n + 1  # Shape starts at 0
    # TODO: Ne GRES po vseh
    for index_pair, value in np.ndenumerate(a):  # Index_pair returns tuple (j, k); a is a_jk
        print(index_pair)
        j = index_pair[0]
        k = index_pair[1]
        alpha = -np.sign(a[k+1][k]) * np.sqrt(np.sum([(a[i][k])**2 for i in range(k + 1, dim - 1)]))
        r = np.sqrt(0.5*(alpha**2 - a[k+1][k]*alpha))
        # TODO: Condense vector v generation?
        v = []
        for l in range(n):
            if l < k:
                v.append(0)
            elif l == k:
                v.append((a[k+1][k] - alpha)/(2*r))
            elif l > k:
                v.append((a[l][k])/(2*r))
            else:
                raise IndexError("Shit's fucky yo!")
        vect = np.array(v)
        P = np.eye(n) - 2*np.outer(vect, vect)
        # print("Pair: {}\nValue:{}".format(index_pair, a))
    return np.matmul(P.T, np.matmul(a, P))