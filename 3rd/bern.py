import numpy as np
import matplotlib.pyplot as plt
import math
import random

def bern(mu):
    """
    distribution following bern
    Parameters
    -----------



    Returns
    -----------


    """

    if random.randrange(1, 101, 1) <= mu * 100:
        return 1
    else:
        return 0


def beta(mu, a, b):
    """
    # muである確率が帰ってくる
    """

    B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    print(B)

    return (mu ** (a - 1)) * ((1 - mu) ** (b - 1)) / B

def main():
    """
    """

    # 初期値
    a = 0.5
    b = 0.7

    # サンプル数
    N = 50
    sample_datas = []

    # trueの値
    mu = 0.25

    for i in range(N):
        sample_datas.append(bern(mu))
        # sampling
        if i % 10 == 0:
            # update mu
            a = sum(sample_datas) + a
            b = i - sum(sample_datas) + b
            mu_probs = [] 
            
            for j in np.arange(0, 1, 0.01):
                mu_probs.append(beta(j, a, b))

            plt.plot(np.arange(0, 1, 0.01), mu_probs)
            plt.show()

if __name__ == "__main__":
    main()