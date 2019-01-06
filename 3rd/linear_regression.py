import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy import stats

class BayezianRegression():
    """
    linear regression model by using bayezian estimation

    Attributes
    ------------
    M : int
        model dimention
    ram : float
        precition, parameter of model, inverse of variance
    """
    def __init__(self, M):
        self.M = M # モデルの次元
        self.ram = 10. # モデルの精度（既知）

        # 事前分布のパラメータ
        self.pre_m = np.zeros(self.M) # 平均
        self.pre_co_precision = np.diag(np.eye(self.M)) # 共分散行列

        # 事後分布のパラメータ
        self.m = np.zeros(self.M) # 平均
        self.co_precision = np.diag(np.ones(self.M)) # 共分散行列

        # 予測分布のパラメータ
        self.mu_opt = None
        self.ram_inv_opt = None

    def fit(self, X_data, Y_data):
        """
        fit the model by using input data and output data

        Parameters
        -----------
        X_data : numpy.ndarray, shape(N, M)
        Y_data : numpy.ndarray, shape(N, )
        
        Returns
        ---------

        
        """
        self.co_precision = self.ram * np.dot(X_data.T, X_data) + self.co_precision
        print(" np.dot(X_data.T, X_data) = {0}".format( np.dot(X_data.T, X_data)))

        temp_1 = np.linalg.inv(self.co_precision)
        temp_2 = self.ram * (np.dot(Y_data, X_data) + np.dot(self.co_precision, self.m))
        print("temp_1 = {0}".format(temp_1))
        print("temp_2 = {0}".format(temp_2))

        self.m = np.dot(temp_1, temp_2.reshape(-1, 1))

        print("m = {0}".format(self.m))

    def predict_distribution(self, X_data):
        """
        make the distribution
        
        Parameters
        ------------
        X_data : numpy.ndarray, shape(M, )

        Returns
        ---------
        
        """
        self.mu_opt = np.dot(self.m.flatten(), X_data.reshape(-1, 1))

        temp_1 = np.linalg.inv(self.co_precision)
        self.ram_inv_opt = 1. / self.ram + np.dot(X_data, np.dot(temp_1, X_data.reshape(-1, 1)))
        deviation = math.sqrt(self.ram_inv_opt)

        return self.mu_opt, self.ram_inv_opt, deviation

    def test_distribution(self):
        """
        """
        # wを計算
        w = np.random.multivariate_normal(self.m, np.linalg.inv(self.co_precision))

        sample_ys = []
        true_ys = []

        for sample_x in np.arange(-1, 1, 0.05):
            # sampling
            xs = [1.]
            for _ in range(self.M-1):
                xs.append(xs[-1] * sample_x)

            xs = np.array(xs)
            
            m = np.dot(w, xs.reshape(-1, 1))
            ram_inv = 1./ self.ram
            y = np.random.multivariate_normal(m, [[ram_inv]])
            sample_ys.append(y)
            true_ys.append(m)

        return w, true_ys, sample_ys

def main():
    # test
    dim = 10
    regressioner = BayezianRegression(dim)
    # dataをサンプルする
    w, true_ys, sample_ys = regressioner.test_distribution()

    plt.scatter(np.arange(-1, 1, 0.05), sample_ys, color="orange")
    plt.plot(np.arange(-1, 1, 0.05), true_ys)
    plt.show()

    # make data set
    X_data = []
    Y_data = []
    for _ in range(10):
        x_1 = (7.5 + 1.) * np.random.rand() + (- 1.)
        y = math.sin(x_1)
        temp_xs = [1.]
        for _ in range(dim-1):
            temp_xs.append(temp_xs[-1] * x_1)
        
        X_data.append(temp_xs)
        Y_data.append(y)
    
    # to numpy
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)

    # fit the model
    regressioner.fit(X_data, Y_data)

    # prediction 
    mus = []
    devs = []
    
    for sample_x in np.arange(-1., 7.5, 0.1):
        # make X_data
        x_data = [1.]
        for _ in range(dim-1):
            x_data.append(x_data[-1] * sample_x)

        mu_opt, ram_inv_opt, deviation = regressioner.predict_distribution(np.array(x_data))

        mus.append(mu_opt)
        devs.append(deviation)
    
    mus = np.array(mus).flatten()
    devs = np.array(devs)

    print(mus)

    plt.plot(np.arange(-1., 7.5, 0.1), mus, color="orange")
    plt.plot(np.arange(-1., 7.5, 0.1), mus - devs, linestyle="dashed")
    plt.plot(np.arange(-1., 7.5, 0.1), mus + devs, linestyle="dashed")
    plt.scatter(X_data[:, 1], Y_data)
    plt.show()

if __name__ == "__main__":
    main()