import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn import neighbors
from sklearn import datasets

from sklearn.metrics import recall_score
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from scipy.stats import multivariate_normal


""" ============== Load Data and separate the Training part and Target ============ """

def Loaddata(CrabTotal):
    Craba = []
    Crabb = []
    d = len(CrabTotal[:])
    f = len(CrabTotal[0]) - 1
    for i in range(d):
        if CrabTotal[:, f][i] == 1:
            Craba.append(list(CrabTotal[i, :f]))
        else:
            Crabb.append(list(CrabTotal[i, :f]))
    return [Craba, Crabb]

""" =======================  Probabilistic  Generative Models=======================
                   The first two parameters are the training data
                                the X is the test data                              """
def ProGen(CrabA, CrabB, CrabTotal, X):

    meanA = np.mean(CrabA, axis=0)
    meanB = np.mean(CrabB, axis=0)
    covA = np.cov(np.array(CrabA).T)
    covB = np.cov(np.array(CrabB).T)
    pA = len(CrabA) / len(CrabTotal)
    pB = len(CrabB) / len(CrabTotal)
    yA = multivariate_normal.pdf(X, mean=meanA, cov=covA, allow_singular='False')
    yB = multivariate_normal.pdf(X, mean=meanB, cov=covB, allow_singular='False')
    NomiA = pA * yA
    NomiB = pB * yB
    Denomi = NomiA + NomiB
    pA1 = NomiA / Denomi
    Zero = []
    One = []
    l2 = []
    for i in range(len(yA)):
        if pA1[i] < 0.5:
            Zero.append(i)
            l2.append(0)
        else:
            One.append(i)
            l2.append(1)
    print(len(Zero), pA1)
    return [Zero, One, pA1, l2]

""" ======================= K_NN with Crossing Validation============================
                    The first two parameters are the training data and the label
                                the X is the test data                                      """

def K_NN(Xtrain00, Xtrain01, X00, q):
    k_range = range(1, 20)                                         # the range of K
    cv_scores = []                                                 # the average accuracy when different K
    for n in k_range:
        knn = KNeighborsClassifier(n)
        scores = cross_val_score(knn, Xtrain00, Xtrain01, cv=5,    # Crossing Validation  cv is the number of folds
                                 scoring='accuracy')
        cv_scores.append(scores.mean())
    cvMax = max(cv_scores)                                         # Choosing the biggest one and indexing its K
    kOptimiz = cv_scores.index(cvMax) + 1
    plt.plot(k_range, cv_scores)                                   # plotting the picture between accuracy and K
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()

    knn = KNeighborsClassifier(kOptimiz)
    d = len(X00[:,0])
    X01 = X00[:,:q]
    l0 = []
    l1 = []
    knn.fit(Xtrain00, Xtrain01)
    for i in range(d):
        if knn.predict(list([X01[i]])) == [[0]]:
            l0.append(i)
        else:
            l1.append(i)
    return [l0, l1]


""" =======================  Load Test Data for Crab==================================== """

CrabTotalTest = np.loadtxt('CrabDatasetforTest.txt')
X0 = CrabTotalTest[:, :7]
X1 = CrabTotalTest[:, 7:]

TenTotalTest = np.loadtxt('10dDatasetforTest.txt')
TX0 = TenTotalTest[:, :10]
TX1 = TenTotalTest[:, 10:]
""" ======================== Calling function =========================== """

CrabTotal0 = np.loadtxt('CrabDatasetforTrain.txt')
Xtrain0 = CrabTotal0[:, :7]
Xtrain1 = CrabTotal0[:, 7:]

TenTotal0 = np.loadtxt('10dDatasetforTrain.txt')
Ttrain0 = TenTotal0[:, :10]
Ttrain1 = TenTotal0[:, 10:]

data = Loaddata(CrabTotal0)
data0 = Loaddata(TenTotal0)

# qaz = ProGen(data[0], data[1], CrabTotal0, X0)
qaz0 = ProGen(data0[0], data0[1], TenTotal0, TX0)
a000 = K_NN(Xtrain0, Xtrain1, X0, 7)
a001 = K_NN(Ttrain0, Ttrain1, TX0, 10)

""" ========================  Plot Results ============================== """

y_true = TX1
y_scores = qaz0[2]
y_pred = qaz0[3]
precision, recall, thresholds = metrics.roc_curve(y_true, y_scores)
plt.xlabel('FTR')
plt.ylabel('TPR')
plt.plot(precision, recall)
plt.show()


print(confusion_matrix(y_true,y_pred))
tp,fp,fn,tn=confusion_matrix(y_true,y_pred).ravel()
print(tp,fp,fn,tn)


guess = y_true
fact = y_pred

classes = list(set(fact))
classes.sort()
confusion = confusion_matrix(guess, fact)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('guess')
plt.ylabel('fact')
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index])

plt.show()


