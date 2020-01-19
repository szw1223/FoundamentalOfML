import numpy as np
from scipy import stats

x = np.mat([[2, 3, 3, 4, 5, 7], [2, 4, 5, 5, 6, 8]])
A = x.T @ x
B = x @ x.T

eigenvaluesa, eigenvectorsa = np.linalg.eig(A)
eigenvaluesb, eigenvectorsb = np.linalg.eig(B)

c = np.mean(x, axis=1)
f = x - c

C = f @ f.T
eigenvaluesc, eigenvectorsc = np.linalg.eig(C)

cd = [-0.66451439, -0.74727547]
D = np.dot(cd, x)

# print(D, eigenvaluesc)

def EM(pi, p, q, MaximumNumberOfIterations):
    test = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    mulst0 = []
    for i, t in enumerate(test):
        nomi0 = pi * p ** t
        nomi1 = (1 - p) ** (1 - t)
        denomi0 = (1 - pi) * q ** t
        denomi1 = (1 - q) ** (1 - t)
        mu = nomi0 * nomi1 / (nomi0 * nomi1 + denomi0 * denomi1)
        mulst0.append(mu)

    NumberIterations = 1

    while NumberIterations <= MaximumNumberOfIterations:
        mulst1 = mulst0 if NumberIterations == 1 else mulst1[10: 20]
        print(mulst1)
        musum = sum(mulst1)
        pi = musum / 10
        # p, q
        p0, q0, q1 = [], [], []
        p1 = musum

        for i, t in enumerate(mulst1):
            p0.append(t * test[i])
            q0.append((1 - t) * test[i])
            q1.append(1 - t)
        p = sum(p0) / p1
        q = sum(q0) / sum(q1)
        # Solve for mu

        for i, t in enumerate(test):
            nomi0 = pi * p ** t
            nomi1 = (1 - p) ** (1 - t)
            denomi0 = (1 - pi) * q ** t
            denomi1 = (1 - q) ** (1 - t)
            mu = nomi0 * nomi1 / (nomi0 * nomi1 + denomi0 * denomi1)
            mulst1.append(mu)

        NumberIterations = NumberIterations + 1
        print(pi, p, q)
    print(pi, p, q, mulst1[10: 20])
    return pi, p, q, mulst1[10: 20]

# EM(0.5, 0.5, 0.5, 1)
# EM(0.5, 0.5, 0.5, 2)
# EM(0.4, 0.6, 0.7, 2)
EM(0.5, 0.5, 0.5, 4)
EM(0.4, 0.6, 0.7, 20)








