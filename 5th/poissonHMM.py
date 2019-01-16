import numpy as np
from scipy import stats
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
        

    def fit(self, X):
        """
        """
        
        # for q(Sn)


def main():
    """
    """
    # jld load data
    f = h5py.File("timeseries.jld", "r")
    data = f["obs"][()]

    plt.plot(range(len(data)), data)
    plt.show()

if __name__ == "__main__":
    main()


    