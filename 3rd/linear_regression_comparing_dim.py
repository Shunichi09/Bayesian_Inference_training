import numpy as np
import matplotlib.pyplot as plt
import math
import random

from animation import AnimDrawer

class BayezianRegression():
    """
    linear regression model by using bayezian estimation

    Attributes
    ------------
    M : int
        model dimention
    lam : float
        precision, parameter of model, inverse of variance
    pre_m : float
        mean
    pre_co_precision : float
        precision
    m : numpy.ndrray
        mean
    co_precision : numpy.ndarray
        precision
    m_opt :  numpy.ndarray
        estimated mean
    lam_inv_opt : numpy.ndarray
        estimated variance
    """
    def __init__(self, M):
        self.M = M # モデルの次元
        self.lam = 10. # モデルの精度（既知）

        # 事前分布のパラメータ
        self.pre_m = np.zeros(self.M) # 平均
        self.pre_co_precision = np.diag(np.eye(self.M)) # 共分散行列

        # 事後分布のパラメータ
        self.m = np.zeros(self.M) # 平均
        self.co_precision = np.diag(np.ones(self.M)) # 共分散行列

        # 予測分布のパラメータ
        self.m_opt = None
        self.lam_inv_opt = None

    def fit(self, X_data, Y_data):
        """
        fit the model by using input data and output data

        Parameters
        -----------
        X_data : numpy.ndarray, shape(N, M)
        Y_data : numpy.ndarray, shape(N, )        
        """

        self.co_precision = self.lam * np.dot(X_data.T, X_data) + self.co_precision
        print(" np.dot(X_data.T, X_data) = {0}".format( np.dot(X_data.T, X_data)))

        temp_1 = np.linalg.inv(self.co_precision)
        temp_2 = self.lam * (np.dot(Y_data, X_data) + np.dot(self.co_precision, self.m))
        print("temp_1 = {0}".format(temp_1))
        print("temp_2 = {0}".format(temp_2))

        self.m = np.dot(temp_1, temp_2.reshape(-1, 1))

        print("m = {0}".format(self.m))

    def predict_distribution(self, X_data):
        """
        make the distributionBayes
        Bayes
        Parameters
        ------------
        X_data : numpy.ndarray, shapeBayes(M, )

        Returns
        ---------
        m_opt : float
            estimated mean
        lam_inv_opt : float
            estimated variance
        deviation : float
            estimated deviation
        """
        self.m_opt = np.dot(self.m.flatten(), X_data.reshape(-1, 1))

        temp_1 = np.linalg.inv(self.co_precision)
        self.lam_inv_opt = 1. / self.lam + np.dot(X_data, np.dot(temp_1, X_data.reshape(-1, 1)))
        deviation = math.sqrt(self.lam_inv_opt)

        return self.m_opt, self.lam_inv_opt, deviation

    def test_distribution(self):
        """test
        This is test program for if it can make the gauss distribution
        Returns
        ---------
        w : numpy.ndarray
            parameter of linear regression
        true_ys : numpy.ndarray
            observation value made without noise
        sample_ys : numpy.ndarray
            observation value made with noise
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
            lam_inv = 1./ self.lam
            y = np.random.multivariate_normal(m, [[lam_inv]])
            sample_ys.append(y)
            true_ys.append(m)

        return w, true_ys, sample_ys

def main():
    dim = 2 # dimention of the linear regression model
    regressioner = BayezianRegression(dim) # make model
    
    """test
    # dataをサンプルする
    w, true_ys, sample_ys = regressioner.test_distribution()
    plt.scatter(np.arange(-1, 1, 0.05), sample_ys, color="orange")
    plt.plot(np.arange(-1, 1, 0.05), true_ys)
    plt.show()
    """

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

        mu_opt, lam_inv_opt, deviation = regressioner.predict_distribution(np.array(x_data))

        mus.append(mu_opt)
        devs.append(deviation)
    
    mus = np.array(mus).flatten()
    devs = np.array(devs)

    # figure
    figure = plt.figure()
    axis_1 = figure.add_subplot(211)
    axis_2 = figure.add_subplot(212)

    axis_1.plot(np.arange(-1., 7.5, 0.1), mus, color="orange")
    axis_1.plot(np.arange(-1., 7.5, 0.1), mus - devs, linestyle="dashed")
    axis_1.plot(np.arange(-1., 7.5, 0.1), mus + devs, linestyle="dashed")
    axis_1.scatter(X_data[:, 1], Y_data)
    
    dim = 7 # dimention of the linear regression model
    regressioner = BayezianRegression(dim) # make model

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

        mu_opt, lam_inv_opt, deviation = regressioner.predict_distribution(np.array(x_data))

        mus.append(mu_opt)
        devs.append(deviation)
    
    mus = np.array(mus).flatten()
    devs = np.array(devs)
    
    axis_2.plot(np.arange(-1., 7.5, 0.1), mus, color="orange")
    axis_2.plot(np.arange(-1., 7.5, 0.1), mus - devs, linestyle="dashed")
    axis_2.plot(np.arange(-1., 7.5, 0.1), mus + devs, linestyle="dashed")
    axis_2.scatter(X_data[:, 1], Y_data)
    
    plt.show()

if __name__ == "__main__":
    main()