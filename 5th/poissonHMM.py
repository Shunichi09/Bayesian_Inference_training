import numpy as np
from scipy import stats
from scipy.special import digamma, logsumexp
import matplotlib.pyplot as plt
import copy

import h5py

class PoissonHMM():
    """
    Attributes
    ------------


    """
    def __init__(self):
        """
        """
        self.class_num = 2

        # Hyper parameters
        # matrix of transition probability
        self.init_A = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.A = None

        # transition probability param
        self.beta = 100.0 * np.eye(self.class_num) + 1.0 * np.ones((self.class_num,self.class_num))
        self.init_beta = copy.deepcopy(self.beta)

        # first probability
        self.pi = np.array([1./2., 1./2.])

        # first probability param
        self.init_alpha = 10.0 * np.ones(self.class_num)
        self.alpha = copy.deepcopy(self.init_alpha)
        
        # lam
        self.lam_sample = np.ones(self.class_num) 

        # param of lam
        self.init_a = np.ones(self.class_num) * 10.
        self.a = copy.deepcopy(self.init_a)
        self.init_b = np.ones(self.class_num) * 10.
        self.b = copy.deepcopy(self.init_b)

    def _init_params(self):
        # want estimated param
        self.zeta = np.zeros((self.data_num, self.class_num)) # shape(data_num, class)
        for n in range(self.data_num):
            self.zeta[n, :] = np.random.dirichlet(np.ones(self.class_num)/self.class_num, 1)

        # self.s = np.zeros((self.data_num, self.class_num)) # shape(data_num, class)
        # for n in range(self.data_num):
        #     self.s[n, :] = np.random.dirichlet(np.ones(self.class_num)/self.class_num, 1)
        # print("s = {0}".format(self.s))

    def fit(self, X, iteration_num = 100):
        """
        Parameters
        ------------
        X : numpy.ndarray, shape(data_num, )
        """
        self.data_num = len(X)
        self._init_params()

        for _ in range(iteration_num):
            
            ln_lkh = np.zeros((self.data_num, self.class_num)) # In p(x_n | lambda_i)

            for k in range(self.class_num):
                ln_lam = digamma(self.a[k]) - np.log(self.b[k]) # <In lambda_i>
                lam = self.a[k] / self.b[k] # <lambda_i>
                for n in range(self.data_num):
                    ln_lkh[n, k] = np.sum(X[n] * ln_lam - lam) 

            expt_ln_pi = digamma(self.alpha) - digamma(np.sum(self.alpha)) # <In pi>

            expt_ln_A = np.zeros((self.class_num,self.class_num)) # <In A_j_i>
            for k in range(self.class_num): # jまで計算する必要ない（digamma がやってくれる）
                expt_ln_A[:,k] = digamma(self.beta[:,k]) - digamma(np.sum(self.beta[:,k]))
            
            # n = 0
            ln_expt_zeta = np.log(self.zeta+1e-10) # 値が飛ばないように（参考：http://d.hatena.ne.jp/echizen_tm/20100628/1277735444

            ln_expt_zeta[0, :] = ln_lkh[0, :] + expt_ln_pi + np.dot(expt_ln_A.T, np.exp(ln_expt_zeta[1, :]))
            ln_expt_zeta[0, :] -= logsumexp(ln_expt_zeta[0, :])

            # 0 < n < N-1
            for n in range(1, self.data_num-1):
                ln_expt_zeta[n, :] = ln_lkh[n, :] + np.dot(expt_ln_A, np.exp(ln_expt_zeta[n-1, :])) + np.dot(expt_ln_A.T, np.exp(ln_expt_zeta[n+1, :]))
                ln_expt_zeta[n, :] -= logsumexp(ln_expt_zeta[n, :])

            # n = N-1
            ln_expt_zeta[self.data_num-1, :] = ln_lkh[self.data_num-1, :] + np.dot(expt_ln_A, np.exp(ln_expt_zeta[self.data_num-2, :]))
            ln_expt_zeta[self.data_num-1, :] -= logsumexp(ln_expt_zeta[self.data_num-1, :])

            expt_zeta = np.exp(ln_expt_zeta) # 最後にまとめてexpを取る
    
            expt_zetazeta = np.zeros((self.data_num, self.class_num, self.class_num))
            for n in range(self.data_num-1):
                for j in range(self.class_num):
                    for s in range(self.class_num):
                        expt_zetazeta[n+1, j, s] = expt_zeta[n+1, j] * expt_zeta[n, s] # 教科書とは順番入れ替えてるだけ

            # update parameters
            # for Gam lamda 各クラスのパラメータ
            sum_zeta = np.sum(expt_zeta, axis=0).flatten()
            print("sum zeta = {0}".format(sum_zeta))
            # print("zeta = {0}".format(expt_zeta))
            # input()
            
            temp = np.dot(expt_zeta.T, X.reshape(-1, 1))
            print("sum zeta = {0}".format(temp))
            # input()
            mul_zx = temp.flatten()

            for k in range(self.class_num):
                self.a[k] = mul_zx[k] + self.init_a[k]
                self.b[k] = sum_zeta[k] + self.init_b[k]

            # for beta 状態遷移行列のパラメータ
            sum_zetazeta = np.sum(expt_zetazeta, axis=0)

            self.beta = sum_zetazeta + self.init_beta

            # for ディリクレ分布
            self.alpha = expt_zeta[0, :] + self.init_alpha

            print(self.init_a)
            print(self.init_b)
            print(self.init_alpha)
            print(self.init_beta)

            self.zeta = copy.deepcopy(expt_zeta)

        return expt_zeta

    def _init_params_test(self):
        # want estimated param
        self.zeta = np.zeros((self.class_num, self.data_num)) # shape(data_num, class)
        for n in range(self.data_num):
            self.zeta[:, n] = np.random.dirichlet(np.ones(self.class_num)/self.class_num, 1)

        # self.s = np.zeros((self.data_num, self.class_num)) # shape(data_num, class)
        # for n in range(self.data_num):
        #     self.s[n, :] = np.random.dirichlet(np.ones(self.class_num)/self.class_num, 1)
        # print("s = {0}".format(self.s))


    def fit_test(self, X, iteration_num = 100):
        """
        Parameters
        ------------
        X : numpy.ndarray, shape(data_num, )
        """
        self.data_num = len(X)
        self._init_params_test()

        for _ in range(iteration_num):
            
            ln_lkh = np.zeros((self.class_num, self.data_num)) # In p(x_n | lambda_i)

            for k in range(self.class_num):
                ln_lam = digamma(self.a[k]) - np.log(self.b[k]) # <In lambda_i>
                lam = self.a[k] / self.b[k] # <lambda_i>
                for n in range(self.data_num):
                    ln_lkh[k, n] = np.sum(X[n] * ln_lam - lam) 

            expt_ln_pi = digamma(self.alpha) - digamma(np.sum(self.alpha)) # <In pi>

            expt_ln_A = np.zeros((self.class_num,self.class_num)) # <In A_j_i>
            for k in range(self.class_num): # jまで計算する必要ない（digamma がやってくれる）
                expt_ln_A[:,k] = digamma(self.beta[:,k]) - digamma(np.sum(self.beta[:,k]))
            
            # n = 0
            ln_expt_zeta = np.log(self.zeta + 1e-10) # 値が飛ばないように（参考：http://d.hatena.ne.jp/echizen_tm/20100628/1277735444
            
            """
            print("zeta = {0}".format(self.zeta))
            input()
            print("ln zeta = {0}".format(ln_expt_zeta))
            input()
            """

            ln_expt_zeta[:, 0] = ln_lkh[:, 0] + expt_ln_pi + np.dot(expt_ln_A.T, np.exp(ln_expt_zeta[:, 1]))
            ln_expt_zeta[:, 0] -= logsumexp(ln_expt_zeta[:, 0])

            # 0 < n < N-1
            for n in range(1, self.data_num-1):
                ln_expt_zeta[:, n] = ln_lkh[:, n] + np.dot(expt_ln_A, np.exp(ln_expt_zeta[:, n-1])) + np.dot(expt_ln_A.T, np.exp(ln_expt_zeta[:, n+1]))
                ln_expt_zeta[:, n] -= logsumexp(ln_expt_zeta[:, n])

            # n = N-1
            ln_expt_zeta[:, self.data_num-1] = ln_lkh[:, self.data_num-1] + np.dot(expt_ln_A, np.exp(ln_expt_zeta[:, self.data_num-2]))
            ln_expt_zeta[:, self.data_num-1] -= logsumexp(ln_expt_zeta[:, self.data_num-1])

            expt_zeta = np.exp(ln_expt_zeta) # 最後にまとめてexpを取る
    
            expt_zetazeta = np.zeros((self.class_num, self.class_num, self.data_num))
            for n in range(self.data_num-1):
                for j in range(self.class_num):
                    for s in range(self.class_num):
                        expt_zetazeta[j, s, n+1] = expt_zeta[j, n+1] * expt_zeta[s, n] # 教科書とは順番入れ替えてるだけ

            # update parameters
            # for Gam lamda 各クラスのパラメータ
            sum_zeta = np.sum(expt_zeta, axis=1)

            """
            print("sum zeta = {0}".format(sum_zeta))
            print("zeta = {0}".format(expt_zeta))
            print("expt_ln_pi = {0}".format(expt_ln_pi))
            print("expt_ln_A = {0}".format(expt_ln_A))
            print("expt_ln_lkh = {0}".format(ln_lkh))
            input()
            print("a = {0}".format(self.a))
            print("b = {0}".format(self.b))
            print("beta = {0}".format(self.beta))
            print("alpha = {0}".format(self.alpha))
            input()
            """
            print("expt_ln_A = {0}".format(np.exp(expt_ln_A)))
            
            mul_zx = np.dot(expt_zeta, X)

            for k in range(self.class_num):
                self.a[k] = mul_zx[k] + self.init_a[k]
                self.b[k] = sum_zeta[k] + self.init_b[k]

            # for beta 状態遷移行列のパラメータ
            sum_zetazeta = np.sum(expt_zetazeta, axis=2) #

            self.beta = sum_zetazeta + self.init_beta

            # for ディリクレ分布
            self.alpha = expt_zeta[:, 0] + self.init_alpha

            self.zeta = copy.deepcopy(expt_zeta)

        return expt_zeta


def main():
    """
    """
    # jld load data
    f = h5py.File("timeseries.jld", "r")
    data = f["obs"][()]

    N = 500 # number of samples
    D = 1 # nubmer of dimensions
    K = 2 # number of clusters

    # poisson distribution, observed data
    a_model = 1.0 * np.ones((D,K))
    b_model = 0.01 * np.ones(K)

    # dirichlet distribution, phi
    alpha_model = 25.0 * np.ones(K)

    # dirichlet distribution, A
    beta_model = 50.0 * np.eye(K) + 1.0 * np.ones((K,K))

    lam_true = np.zeros((D,K))
    for k in range(K):
        lam_true[:,k] = np.random.gamma(a_model[:,k], scale=1/b_model[k])
    phi_true = np.random.dirichlet(alpha=alpha_model)
    A_true = np.zeros((K,K))
    for i in range(K):
        A_true[:,i] = np.random.dirichlet(alpha=beta_model[:,i])

    S_true = np.zeros((K,N))
    for n in range(N):
        if n == 0:
            S_true[:,n] = np.random.multinomial(n=1, pvals=phi_true)
        else:
            s = np.argmax(S_true[:,n-1])
            S_true[:,n] = np.random.multinomial(n=1, pvals=A_true[:,s])

    X = np.zeros((D,N))
    for n in range(N):
        s = np.argmax(S_true[:,n])
        for d in range(D):
            X[d,n] = np.random.poisson(lam=lam_true[d,s])

    data = X.flatten()

    figure = plt.figure()
    axis1 = figure.add_subplot(211)
    axis2 = figure.add_subplot(212)

    axis1.plot(range(len(data)), data)

    # make model
    model = PoissonHMM()

    # zeta = model.fit(data)

    zeta = model.fit_test(data)

    # print("zeta = {0}".format(zeta))

    axis2.plot(range(len(data)), zeta[1, :])
    axis2.fill_between(range(len(data)), zeta[1, :], np.zeros(len(data)), facecolor='b', alpha=0.5)

    plt.show()
    

if __name__ == "__main__":
    main()


    