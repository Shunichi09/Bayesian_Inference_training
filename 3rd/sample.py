import math
import numpy as np
from pylab import *

def beta(theta, a, b):
    """
    beta distribution

    theta^{alpha-1} * (1-theta)^{beta-1}
    ------------------------------------
               B(alpha, beta)

    B(alpha, beta)
      gamma(a) * gamma(b)
    = -------------------
        gamma(a + b)
    """
    B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    return (theta ** (a - 1)) * ((1 - theta) ** (b - 1)) / B

if __name__ == '__main__':
    thetas = np.arange(0, 1, 0.01)
    params = [(0.5, 0.5), (1., 0.5), (2., 0.5), (3., 0.5), (4., 0.5),
              (0.5, 1.), (1., 1.), (2., 1.), (3., 1.), (4., 1.),
              (0.5, 2.), (1., 2.), (2., 2.), (3., 2.), (4., 2.),
              (0.5, 3.), (1., 3.), (2., 3.), (3., 3.), (4., 3.),
              (0.5, 4.), (1., 4.), (3., 4.), (3., 4.), (4., 4.)
             ]

    axisnum = 0
    for (a, b) in params:
        axisnum += 1
        subplot(5, 5, axisnum)
        betas = [beta(theta, a, b) for theta in thetas]
        plot(thetas, betas)
        xlim(0, 1.)
        ylim(0, 3.)
        text(0.5, 2.7, 'a=%.1f b=%.1f' % (a, b),
                horizontalalignment='center', verticalalignment='center')
        xlabel('theta')
        ylabel('p(theta)')
    show()