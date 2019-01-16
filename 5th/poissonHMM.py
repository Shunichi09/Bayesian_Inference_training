import numpy as np
from scipy import stats
from scipy.special import digamma, logsumexp, gammaln
import matplotlib.pyplot as plt

import h5py

def make_dataset(nums=500):
    """making time series dataset
    Parameters
    ------------
    nums : int

    Returns
    ----------
    samples : numpy.ndarray
    """
    
    return nums

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
        self.beta = np.random.randint(5, 10, (self.class_num)) * 1.
        self.init_beta = np.random.randint(5, 10, (self.class_num)) * 1.

        # first probability
        self.pi = np.array([1./2., 1./2.])

        # first probability param
        self.alpha = np.random.randint(5, 10, (self.class_num)) * 1.
        self.init_alpha = np.random.randint(5, 10, (self.class_num)) * 1.

        # lam
        self.lam_sample = np.ones(self.class_num) 

        # param of lam
        self.init_a = np.ones(self.class_num) * 2.
        self.a = copy.deepcopy(self.init_a)
        self.init_b = np.ones(self.class_num) * 2.
        self.b = copy.deepcopy(self.init_b)

        # want estimated param
        self.eata = None # shape(data_num, class)
        self.s = None # shape(data_num)

    def fit(self, X, iteration_num = 1000):
        """
        Parameters
        ------------
        X : numpy.ndarray, shape(data_num, )
        """
        self.data_num = len(X)
        for i in range(iteration_num):
            
            for k in range(self.class_num):
                ln_lam = digamma(self.a[k]) - np.log(self.b[k]) # in lamda
                lam = self.a[k] / self.b[k]
                for n in range(self.data_num):
                    ln_lkh[k,n] = np.sum(X[n] * ln_lam - lam)

             expt_ln_phi = digamma(alpha_post) - digamma(np.sum(alpha_post))







        



def main():
    """
    """
    # jld load data
    f = h5py.File("timeseries.jld", "r")
    data = f["obs"][()]

    figure = plt.figure()
    axis1 = figure.add_subplot(211)
    axis2 = figure.add_subplot(212)

    axis1.plot(range(len(data)), data)
    



if __name__ == "__main__":
    main()


    