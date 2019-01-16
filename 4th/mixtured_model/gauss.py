import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import math
from scipy import stats
from sklearn.utils import shuffle

from animation import AnimDrawer

def draw_elipse(m, co, kai_val=5.991):
    """make elipse with mean and co-varince matrix
    Parameters
    -----------
    m : numpy.ndarray
        mean
    co : numpy.ndarray
        covariance matrix
    kai_val : float
        kai square distribution
    Returns
    ------------
    e_xs : numpy.ndarray
        elipse points
    e_ys : numpy.ndarray
        elipse points
    """
    eig_val, eig_vec = np.linalg.eig(co)

    # print("eig_val = {0}".format(eig_val))
    # print("eig_vec = {0}".format(eig_vec))

    a = math.sqrt(np.max(eig_val) * kai_val)
    b = math.sqrt(np.min(eig_val) * kai_val)

    eig_num = np.argmax(eig_val)

    theta = math.atan2(eig_vec[eig_num][1], eig_vec[eig_num][0])

    # sampling
    e_xs = []
    e_ys = []

    for t in np.arange(0, 2. * math.pi, 0.01):
        e_xs.append(a * math.cos(t))
        e_ys.append(b * math.sin(t))

    # rotation
    rot_mat = np.array([[math.cos(theta), math.sin(theta)],
                        [-math.sin(theta), math.cos(theta)]])

    e_points = np.dot(rot_mat, np.array([e_xs, e_ys]))

    e_xs = e_points[0, :] + m[0]
    e_ys = e_points[1, :] + m[1]

    return e_xs, e_ys

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


def sampling_mixture_poisson(nums=100, classes=3):
    """
    """
    sample_points = None
    label = []

    for i in range(classes):

        # make parameter matrix
        eigs = np.random.randint(1, 10, (2))
        long_eig = float(max(eigs))
        short_eig = float(min(eigs))
        eig_mat = np.array([[long_eig, 0.],
                            [0., short_eig]])
        
        theta = abs(np.random.randn())
        rot_mat = np.array([[math.cos(theta), -math.sin(theta)],
                            [math.sin(theta), math.cos(theta)]])
        
        temp_mat = np.dot(rot_mat, np.linalg.inv(eig_mat))
        co_variance_mat = np.dot(temp_mat, rot_mat.T)
        mean = np.array([float(np.random.randint(1, 10)), float(np.random.randint(1, 10))])
        param_mat = np.linalg.inv(co_variance_mat)

        # e_xs, e_ys = draw_elipse(mean, co_variance_mat)
        # plt.plot(e_xs, e_ys)

        print("eig_mat = {0}".format(eig_mat))
        print("mean = {0}".format(mean))
        print("cov = {0}".format(co_variance_mat))
        print("eig = {0}" .format(np.linalg.eig(co_variance_mat)))

        multi_gauss = stats.multivariate_normal(mean, co_variance_mat)

        # sample
        points = multi_gauss.rvs(nums)

        print(points.shape)

        if sample_points is None:
            sample_points = points
        else:
            sample_points = np.vstack((sample_points, points))

        label.extend(np.ones(nums) * (i+1))

    label = np.array(label).flatten()
    sample_points, label = shuffle(sample_points, label)

    plt.scatter(sample_points[:, 0], sample_points[:, 1])
    plt.show()

    return sample_points

class GibbsMixtureGaussModel():
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

    def gibbs_sample(self, iteration_num=100):
        """
        """
        for i in range(iteration_num):

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
    class_num = 2
    data = sampling_mixture_poisson(classes=class_num)

    
    """
    for i, datum in enumerate(data):
        plt.plot(datum[0], datum[1], "o", color=tuple(history_eata_sample[0][i]))
    
    plt.show()

    for i, datum in enumerate(data):
        plt.plot(datum[0], datum[1], "o", color=tuple(history_eata_sample[-1][i]))
    
    plt.show()

    objects = [history_s_sample]
    animdrawer = AnimDrawer(objects, observation_points=data)
    
    animdrawer.draw_anim()
    """

if __name__ == "__main__":
    main()