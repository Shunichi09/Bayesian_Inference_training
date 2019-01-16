import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def main():

    # 事前分布
    a = 2.
    b = 2.
    pre_beta = stats.beta(a, b)

    xs = np.linspace(0., 1.)
    pre_pdf = pre_beta.pdf(xs)

    plt.plot(xs, pre_pdf)
    # plt.show()

    # 事後分布
    a = 1. * 7. + a
    b = 7. - 1. * 7. + b

    af_pdf = stats.beta(a, b)
    af_pdf = af_pdf.pdf(xs)
    plt.plot(xs, af_pdf)
    plt.show()
    
if __name__ == "__main__":
    main()