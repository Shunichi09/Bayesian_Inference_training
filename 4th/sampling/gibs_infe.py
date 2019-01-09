import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
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

def draw_elipse_test(m, co, kai_val=5.991):
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
    e_points : matplotlib.patch.Elipse
        elipse class
    """
    eig_val, eig_vec = np.linalg.eig(co)

    # print("eig_val = {0}".format(eig_val))
    # print("eig_vec = {0}".format(eig_vec))

    a = math.sqrt(np.max(eig_val) * kai_val)
    b = math.sqrt(np.min(eig_val) * kai_val)

    eig_num = np.argmax(eig_val)

    theta = math.atan2(eig_vec[eig_num][1], eig_vec[eig_num][0])

    # print("a = {0}".format(a))
    # print("b = {0}".format(b))

    e_points = Ellipse(m, 2 * a, 2 * b, angle=-math.degrees(theta))

    return e_points

def calc_KL_div(m1, cov1, m2, cov2):
    """calculate KL divergence between Gaussian distributions

    KL|distribution_1||distribution_2|
    """
    sign_1, logdet_1 = np.linalg.slogdet(cov1)
    sign_2, logdet_2 = np.linalg.slogdet(cov2)

    temp = np.trace(np.dot(np.linalg.inv(cov2), cov1))

    d = float(len(m1)) * 1.

    temp_2 = np.dot((m2 - m1), np.dot(np.linalg.inv(cov2), (m2 - m1).reshape(-1, 1)) )

    score = logdet_2 - logdet_1 - d + temp + temp_2

    print("KL div = {0}".format(0.5 * score))

    return 0.5 * score

class GibsSampler():
    """
    Attributes
    -----------

    """
    def __init__(self, m, cov):
        """
        Parameters
        ----------
        m : 

        cov : 

        """
        # estimate mean and covariance
        self.true_mean = m
        self.true_cov = cov

        self.points_x1 = [0.]
        self.points_x2 = [0.]

    def sample(self):
        """
        """
        # sampling x1
        mean = self.true_mean[0] -  1./self.true_cov[0, 0] * self.true_cov[0, 1] * (self.points_x2[-1] - self.true_mean[1])
        point_x1 = np.random.normal(mean, 1./self.true_cov[0, 0])

        self.points_x1.append(point_x1)

        # sampling x2
        mean = self.true_mean[1] -  1./self.true_cov[1, 1] * self.true_cov[1, 0] * (self.points_x1[-1] - self.true_mean[0])
        point_x2 = np.random.normal(mean, 1./self.true_cov[1, 1])

        self.points_x2.append(point_x2)

    def estimate_val(self):
        """
        Returns
        ---------


        """
        # calc mean and covariance 
        mean = np.array([np.mean(self.points_x1), np.mean(self.points_x2)])
        cov = np.cov(np.array([self.points_x1, self.points_x2]))

        return mean, cov

def main():
    sample_num = 1000
    m = np.array([0., 0.])
    cov = np.array([[1., 0.0], [0.0, 1.]])

    sampler = GibsSampler(m, cov)

    history_means = []
    history_covs = []
    history_KL_score = []

    for i in range(sample_num):
        sampler.sample()

        if i > 5:
            est_m, est_cov = sampler.estimate_val()
            history_means.append(est_m)
            history_covs.append(est_cov)

            KL_score = calc_KL_div(est_m, est_cov, m, cov)
            history_KL_score.append(KL_score)

    observation_points = np.array([sampler.points_x1[:sample_num-5], sampler.points_x2[:sample_num-5]])
    objects = [observation_points, history_means, history_covs, m, cov, history_KL_score]

    animation = AnimDrawer(objects)
    animation.draw_anim()

    plt.plot(range(len(history_KL_score)), history_KL_score)
    plt.show()

    # est_m, est_cov = sampler.estimate_val()
    """
    # true gaussian
    fig = plt.figure()
    
    axis = fig.add_subplot(111)
    axis.set_aspect('equal')

    # points
    axis.scatter(sampler.points_x1[:50], sampler.points_x2[:50])

    e_xs, e_ys = draw_elipse(m, cov)
    est_e_xs, est_e_ys = draw_elipse(est_m, est_cov)
    # e_points = draw_elipse_test(m, co)

    axis.plot(e_xs, e_ys)  
    axis.plot(est_e_xs, est_e_ys)
    plt.show()
    """

if __name__ == "__main__":
    main()
