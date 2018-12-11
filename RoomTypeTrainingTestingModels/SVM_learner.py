from cvxopt import matrix, solvers
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
solvers.options['show_progress'] = False
from sklearn.datasets import make_classification, make_circles

def RBF_kernel(sigma=1):
    def kernel(x, y):
        return np.exp(-np.sum(np.power(x - y,2)) / (2. * sigma))
    return kernel

class KernelSVM(BaseEstimator):
    def __init__(self, C=1, kernel=np.dot):
        self.C = C
        self.kernel = kernel

    def fit(self, X, Y):
        N = len(Y)
        P = matrix([Y[i] * Y[j] * self.kernel(X[i], X[j]) for i in range(N) for j in range(N)], (N, N))
        q = matrix(-np.ones(N))
        G = matrix(np.bmat([[-1. * np.identity(N)], [1. * np.identity(N)]]))
        h = matrix([0.] * N + [self.C] * N)
        A = matrix(1. * Y, (1, N))
        b = matrix(0.)
        sol = solvers.qp(P, q, G, h, A, b)
        self.alpha_ = np.array(sol['x']).flatten()
        self.support_vectors = self.alpha_ > 1e-5
        sv_ind = self.support_vectors.nonzero()[0]
        self.X_sup = X[sv_ind]
        self.Y_sup = Y[sv_ind]
        self.alpha_sup = self.alpha_[sv_ind]
        self.n_sv = len(sv_ind)
        self.bias_ = np.mean([self.Y_sup[i] - np.sum([self.alpha_sup[j] * self.Y_sup[j] * self.kernel(self.X_sup[i], self.X_sup[j]) for j in range(self.n_sv)]) for i in range(self.n_sv)])

    def predict_proba(self, X):
        return [np.sum([self.alpha_sup[i] * self.Y_sup[i] * self.kernel(self.X_sup[i], X[j]) for i in range(self.n_sv)]) + self.bias_ for j in range(len(X))]

    def predict(self, X):
        return np.sign(self.predict_proba(X))

class multiclassSVM:
    def __init__(self,K,C,sigma,kernel):
        self.K = K
        if kernel == "RBF":
            self.models = [KernelSVM(C=C,kernel=RBF_kernel(sigma)) for i in range(K)]
        else:
            self.models = [KernelSVM(C=C,kernel=np.dot) for i in range(K)]

    def train(self,train_data,train_label):
        for k in range(self.K):
            model = self.models[k]
            two_y = np.array(train_label,dtype=float)

            for i in range(len(train_label)):
                if two_y[i] != k:
                    two_y[i] = -1/self.K
                else:
                    two_y[i] = 1

            model.fit(train_data, two_y)

    def pred(self,data):
        result = np.zeros(shape=(self.K,data.shape[0]))
        for k in range(self.K):
            model = self.models[k]
            result[k,:] = model.predict_proba(data)
        result = np.argmax(result,axis=0)
        return result

    def test(self,test_data,test_label):
        result = self.pred(test_data)
        accu = np.sum(result == test_label)/result.shape[0]
        return accu



            # num_class = 5
# X, Y = make_classification(n_samples=100, n_features=5,n_informative= 5, n_redundant=0, n_clusters_per_class=1,n_classes=num_class)
# m = multiclassSVM(5)
# m.train(X,Y)
# accu = m.test(X,Y)
# print(accu)