import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import math
from scipy import stats
from sklearn.utils import shuffle

from animation import AnimDrawer

def calc_softmax(array):
    """
    Parmeters
    ----------
    array : numpy.ndarray
    Returns
    ----------
    normalized_array : numpy.ndarray
    """

    c = np.max(array)
    exp_array = np.exp(array - c)
    sum_exp_array = np.sum(exp_array)

    normalized_array = exp_array / sum_exp_array
    
    return normalized_array


def sampling_mixture_poisson(nums=100, classes=4):
    """
    """
    dim = 2
    sample_points = []
    label = []

    for i in range(classes):
        two_d_points = []
        for _ in range(dim):
            lam =  np.random.randint(10, 300)
            print("lam = {0}".format(lam))
            points = np.random.poisson(lam, (nums))
            two_d_points.append(points)

        sample_points.extend(np.array(two_d_points).T)
        label.extend(np.ones(nums) * (i+1))

    sample_points = np.array(sample_points)
    label = np.array(label).flatten()

    sample_points, label = shuffle(sample_points, label)

    plt.scatter(sample_points[:, 0], sample_points[:, 1])
    plt.show()

    return sample_points

class GibbsMixturedPoisson():
    """
    Attributes
    ----------
    s_sample : numpy.ndarray, shape(N, class_num) 
    lam_sample : numpy.ndarray, shape(class_num, d)
    pi_sample : numpy.ndarray, shape(class_num) 
    alpha_sample : numpy.ndarray, shape(class_num)
    """
    def __init__(self, data, class_num, dim):
        """
        Parameters
        ----------
        class_num : int
        """
        self.data = data
        self.class_num = class_num
        self.dim = dim

        self.s_sample_one_hot = None
        self.s_sample = None
        self.s_sample_E = None

        # Gamma initialize
        self.a_sample = np.random.randint(5, 10, (self.class_num, self.dim)) * 1.
        self.b_sample = np.random.randint(5, 10, (self.class_num, self.dim)) * 1.

        self.init_a_sample = np.random.randint(5, 10, (self.class_num, self.dim)) * 1.
        self.init_b_sample = np.random.randint(5, 10, (self.class_num, self.dim)) * 1.

        self.lam_sample = np.ones((self.class_num, self.dim)) * 50.

        self.alpha_sample = np.random.randint(5, 10, (self.class_num)) * 1.

        self.init_alpha_sample = np.random.randint(5, 10, (self.class_num)) * 1.

        print("lam sample = {0}".format(self.lam_sample))
        input()

        self.pi_sample = np.array([1./3., 1./6., 1./2.])
        # self.pi_sample = np.ones(self.class_num) / float(self.class_num)

        print("pi sample = {0}".format(self.pi_sample))
        input()

        self.history_s_sample = []
        self.history_eata_sample = []

    def gibbs_sample(self):
        """
        """
        self._sample_s()
        self._sample_lam_pi()

        self.history_s_sample.append(self.s_sample)

        print("lam sample = {0}".format(self.lam_sample))
        print("pi sample = {0}".format(self.pi_sample))
        # print(" sample = {0}".format(self.history_s_sample))

        return self.history_s_sample, self.history_eata_sample

    def _sample_s(self):
        """
        """
        temp_s = []
        temp_s_one_hot = []
        temp_eata = []

        for n in range(len(self.data)):
            eata = []

            for k in range(self.class_num):
                temp = 0
                for d in range(self.dim):
                    temp += self.data[n, d] * math.log(self.lam_sample[k, d]) - self.lam_sample[k, d]

                temp += math.log(self.pi_sample[k])
                eata.append(temp)

            # print("eata = {0} sum = {1}".format(eata, sum(eata)))
            
            eata = calc_softmax(np.array(eata))
                
            # print("eata = {0} sum = {1}".format(eata, sum(eata)))
            # input()

            custm = stats.rv_discrete(name='custm', values=(np.arange(self.class_num), eata))
            
            # sampling
            est_class = custm.rvs()
            temp_s.append(est_class)

            temp_eata.append(eata)
            
            temp_one_hot = np.zeros(self.class_num)
            temp_one_hot[est_class] = 1.

            temp_s_one_hot.append(temp_one_hot)
        
        self.s_sample = np.array(temp_s)
        self.s_sample_one_hot = np.array(temp_s_one_hot)
        self.history_eata_sample.append(temp_eata)

    def _sample_lam_pi(self):
        """
        """
        for k in range(self.class_num):

            for d in range(self.dim):

                temp_a = self.init_a_sample[k, d]
                temp_b = self.init_b_sample[k, d]
                temp_alpha = self.init_alpha_sample[k]

                for n in range(len(self.data)):
                    temp_a += self.data[n, d] * self.s_sample_one_hot[n, k]
                    temp_b += self.s_sample_one_hot[n, k]
                    temp_alpha += self.s_sample_one_hot[n, k]

                self.a_sample[k, d] = temp_a
                self.b_sample[k, d] = temp_b

                print("a = {0}".format(self.a_sample))
                print("b = {0}".format(self.b_sample))
                print("alpha = {0}".format(temp_alpha))

                # 分布作成
                self.lam_sample[k, d] = np.random.gamma(temp_a, 1./temp_b)
            
            self.alpha_sample[k] = temp_alpha

        # 分布作成
        self.pi_sample = np.random.dirichlet(self.alpha_sample)

def main():
    dim = 2
    class_num = 3
    data = sampling_mixture_poisson(classes=class_num)

    classifier = GibbsMixturedPoisson(data, class_num, dim)

    iteration_num = 15

    for i in range(iteration_num):
        print("iteration num = {0}".format(i))
        history_s_sample, history_eata_sample = classifier.gibbs_sample()

    colors = ["r", "b", "g", "y"]

    for i, datum in enumerate(data):
        plt.plot(datum[0], datum[1], "o", color=tuple(history_eata_sample[0][i]))
    
    plt.show()

    for i, datum in enumerate(data):
        plt.plot(datum[0], datum[1], "o", color=tuple(history_eata_sample[-1][i]))
    
    plt.show()

    objects = [history_s_sample]
    animdrawer = AnimDrawer(objects, observation_points=data)
    
    animdrawer.draw_anim()

if __name__ == "__main__":
    main()