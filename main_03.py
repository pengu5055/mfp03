import numpy as np
import numpy.linalg as la
import time
import matplotlib.pyplot as plt
import matplotlib.colors as color
import scipy.special
from matplotlib.animation import ArtistAnimation

# TODO: Time functions and compare q and numpy.linalg


def householder_transform_matrix(matrix):  # Najlazje je samo lociti postopek za k = 0 (oz. prvi stolpec matrike)
    """
    Wonky as f and needs debugging because for some odd reason has problems with zero division and returning nan.
    Dejansko tudi, ko dela, dela ocitno tako slabo, da QR potem rabi DLJE da konca.
    """
    a = np.copy(matrix)
    n = np.shape(a)[0]  # Numpy dimenzija (ki je enaka pravi dimenziji, samo indeksi tecejo od 0)
    if wonky_exit(a):
        return a

    alpha = -np.sign(a[1][0]) * np.sqrt(np.sum([a[j][0]**2 for j in range(1, n)]))
    r = np.sqrt(0.5*alpha**2 - 0.5*alpha*a[1][0])
    v = [0, (a[1][0] - alpha)/(2*r)]
    for j in range(2, n):
        v.append(a[j][0]/(2*r))
    vect = np.array(v)
    P = np.eye(n) - 2*np.outer(vect, vect)
    a = np.matmul(np.matmul(P, a), P)
    img.append([plt.imshow(a, cmap="PiYG", norm=color.CenteredNorm(vcenter=0))])
    # img.append([plt.imshow(a, cmap="hot", norm=color.SymLogNorm(linthresh=0.1))])

    for k in range(1, n - 1):  # Ne gre cisto do konca! k-2 => n - 1
        alpha = -np.sign(a[k][k-1]) * np.sqrt(np.sum([a[j][k-1]**2 for j in range(k, n)]))
        r = np.sqrt(0.5*(alpha**2 - a[k][k-1]*alpha))
        v = []
        for l in range(0, n):
            if l < k:
                v.append(0)
            elif l == k:
                v.append((a[k][k-1]-alpha)/(2*r))
            elif l > k:
                v.append((a[l][k-1])/(2*r))
        vect = np.array(v)
        P = np.eye(n) - 2 * np.outer(vect, vect)
        a = np.matmul(P, np.matmul(a, P))
        img.append([plt.imshow(a, cmap="PiYG", norm=color.CenteredNorm(vcenter=0))])
    return a


def zero_filter(matrix, epsilon):
    a = np.copy(matrix)
    for index_pairs, value in np.ndenumerate(a):
        if np.abs(value) < epsilon:
            a[index_pairs[0]][index_pairs[1]] = 0
    img.append([plt.imshow(a, cmap="PiYG", norm=color.CenteredNorm(vcenter=0))])
    return a


def wonky_exit(matrix, epsilon=10**-10):
    return np.abs(np.sum(np.abs(matrix)) - np.sum(np.abs(np.diag(matrix)))) < epsilon


def qr_decomp(matrix):
    a = np.copy(matrix)
    m, n = np.shape(a)
    Q = np.eye(m)
    for i in range(n):
        H = np.eye(m)
        H[i:, i:] = householder_transform_vect(a[i:, i])  # Rabimo spodnji kvadrant oz. kvadrat spodaj
        Q = np.matmul(Q, H)
        a = np.matmul(H, a)
    return Q, a


def householder_transform_vect(vector):
    v = vector/(vector[0] + np.copysign(np.sqrt(np.sum([val**2 for c, val in np.ndenumerate(vector)])), vector[0]))
    v[0] = 1
    H = np.eye(np.shape(vector)[0])
    H -= (2/np.dot(v, v)) * np.outer(v, v)

    return H


def diagonalize(matrix, tol=10**-15, maxiter=10000):
    """
    A diagonalization and eigenvector calculation method
    :param matrix: input matrix to diagonalize
    :param tol: tolerance for zero filter
    :param maxiter: maximum number of iterations
    :return Diagonalized matrix and matrix of eigenvectors
    """
    tridiag = matrix  # DEBUG STEP?
    # tridiag = householder_transform_matrix(matrix)  # Broken for some damn reason
    # tridiag = zero_filter(tridiag, tol)
    s = np.eye(tridiag.shape[0])  # To store eigenvectors
    # print(tridiag)
    for i in range(maxiter):
        # Q = np.eye(tridiag.shape[0])
        print("Iter: {}".format(i))
        if i == maxiter - 1:
            raise Warning("Maximum number of iterations exceeded")
            # warnings.warn("Maximum number of iterations exceeded")
            # break
        if wonky_exit(tridiag):
            print(i)
            break
        Q, R = qr_decomp(tridiag)
        tridiag = zero_filter(np.matmul(Q.T, np.matmul(tridiag, Q)), tol)
        s = np.matmul(s, Q)

    return tridiag, s


def lho(n):
    return np.diag([i + 1/2 for i in range(0, n)])


def delta(i, j):
    return int(i == j)


def q_matrix_single(n):
    matrix = np.zeros((n, n))
    for ind, val in np.ndenumerate(matrix):
        i, j = ind
        matrix[i][j] = 0.5 * np.sqrt(i + j + 1) * delta(np.abs(i - j), 1)

    return np.matmul(matrix, np.matmul(matrix, np.matmul(matrix, matrix)))


def q_matrix_double(n):
    matrix = np.zeros((n, n))
    for ind, val in np.ndenumerate(matrix):
        i, j = ind
        a = np.sqrt(j * (j - 1)) * delta(i, j - 2)
        b = (2 * j + 1) * delta(i, j)
        c = np.sqrt((j + 1) * (j + 2)) * delta(i, j + 2)
        matrix[i][j] = 0.5 * (a + b + c)

    return np.matmul(matrix, matrix)


def q_matrix_quad(n):
    matrix = np.zeros((n, n))
    for ind, val in np.ndenumerate(matrix):
        i, j = ind
        prefac = 1/(2**4) * np.sqrt((2**i * np.math.factorial(i))/(2**j * np.math.factorial(j)))
        a = delta(i, j + 4)
        b = 4*(2 * j + 3) * delta(i, j + 2)
        c = 12*(2*j**2 + 2*j + 1) * delta(i, j)
        d = 16*j*(2*j**2 - 3*j + 1) * delta(i, j - 2)
        e = 16*j*(j**3 - 6*j**2 + 11*j - 6) * delta(i, j - 4)
        matrix[i][j] = prefac * (a + b + c + d + e)

    return matrix


def data_sort(diag, Q):
    diag_elements = np.diag(diag)
    vectors = np.copy(Q)
    output = []
    for coord, val in np.ndenumerate(diag_elements):
        output.append([val, vectors[coord[0]]])
    output.sort()

    return np.array(output)


def basis(q, n):
    return (2**n * np.math.factorial(n) * np.sqrt(np.pi))**(-1/2) * np.exp(-q**2/2) * scipy.special.eval_hermite(n, q)


def anharmonic(lamb, n, func):
    return lho(n) + lamb*func(n)


def plot_poly(x, vect):
    c = [vect[i]*basis(x, i) for i in range(len(vect))]
    return np.sum(c)


def arrayize(array, function, *args):
    return np.array([function(x, *args) for x in array])


def time_diag(x, lamb):  # Idea: x is element of matrix sizes to time diagonalization
    times = []
    for element in list(x):
        data = anharmonic(lamb, element, q_matrix_single)
        pre = time.time()
        diagonalize(data, tol=10**-5)
        post = time.time()
        times.append(post-pre)  # Should return seconds elapsed

    return np.array(times)


def time_eigh(x, lamb):  # Idea: x is element of matrix sizes to time diagonalization
    times = []
    for element in list(x):
        data = anharmonic(lamb, element, q_matrix_single)
        pre = time.time()
        dummy = la.eigh(data)
        post = time.time()
        times.append(post-pre)  # Should return seconds elapsed

    return np.array(times)


fig, ax = plt.subplots()
img = []  # PAZI: To mora biti tu preden klices funkcijo, da ima kam spraviti slike za animacijo!

lam = 0.5
data = anharmonic(lam, 10, q_matrix_single)
# data = anharmonic(lam, 10, q_matrix_double)
# data = anharmonic(lam, 10, q_matrix_quad)

diag, Q = diagonalize(data, tol=10**-5, maxiter=10000)

# Plot matrix heat map animation
ani = ArtistAnimation(fig, img, interval=90, repeat=False, blit=True)
plt.title("Matrix diagonalization")
plt.axis("off")
plt.colorbar()
# ani.save("1.mp4", "ffmpeg", fps=20)
plt.show()

# Plot osnovnih vezanih stanj
# plt.title("Prvih 10 vezanih stanj za nemoten harmonski oscilator")
# x = np.linspace(-5, 5, 200)
# plt.plot(x, x**2)
# plt.ylim(0, 20)
# for i in range(10):
#     plt.plot(x, basis(x, i) + 2*i)
# plt.xlabel(r"$x$")
# plt.ylabel(r"$|n\rangle$")
#
# plt.show()

# Plot lastnih stanj za motnjo lambda
# lam = 1
# plt.title(r"Prvih 10 lastnih stanj za anharmonski oscilator z $\lambda = {}$".format(lam))
# x = np.linspace(-5, 5, 200)
# plt.plot(x, x**2 + lam*x**4, label="V(x)", color="#FFA0FD")
# plt.ylim(-1, 22)
# sort = data_sort(diag, Q)
# i = 0
# color_vec = ["987284", "CDC6AE", "A3B4A2", "38686A", "A18276", "682D63", "414288", "5FB49C", "98DFAF", "CBBAED"]
# for vec in range(Q.shape[1]):
#     plt.plot(x,18 + arrayize(x, plot_poly, Q[:, vec]) - 2*i, color="#{}".format(color_vec[i]))
#     i += 1
# plt.ylabel(r"$|n\rangle$")
# columns = [r"$|n^{0}\rangle$", r"$|n^{1}\rangle$", r"$|n^{2}\rangle$", r"$|n^{3}\rangle$",
#            r"$|n^{4}\rangle$", r"$|n^{5}\rangle$", r"$|n^{6}\rangle$", r"$|n^{7}\rangle$",
#            r"$|n^{8}\rangle$", r"$|n^{9}\rangle$"]
# rows = [r"$E_n$"]
# cell_text = [[round(sort[i][0], 2) for i in range(10)]]
# plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns, bbox=[0, -0.3, 1, 0.2])
# plt.subplots_adjust(bottom=0.21)
# plt.legend()
# plt.show()

# Plot of diagonalization routine run times
# x = np.arange(2, 100)
# y1 = time_diag(x, 0.5)
# y2 = time_eigh(x, 0.5)
# plt.title("Function run times")
# plt.plot(x, y1, color="#D7BCE8", label="diagonalize()")
# plt.plot(x, y2, color="#439A86", label="np.linalg.eigh()")
# plt.xlabel("N")
# plt.yscale("log")
# plt.ylabel("t [s]")
# plt.legend()
# plt.show()
