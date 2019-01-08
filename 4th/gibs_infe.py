import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

def draw_elipse(m, co, kai_val=5.991):
    """make elipse with mean and co-varince matrix
    Parameters
    -----------
    m : numpy.ndarray

    co : numpy.ndarray

    kai_val : float

    Returns
    ------------



    """
    eig_val, eig_vec = np.linalg.eig(co)

    print("eig_val = {0}".format(eig_val))
    print("eig_vec = {0}".format(eig_vec))

    a = math.sqrt(np.max(eig_val) * kai_val)
    b = math.sqrt(np.min(eig_val) * kai_val)

    eig_num = np.argmax(eig_val)

    theta = math.atan2(eig_vec[eig_num][1], eig_vec[eig_num][0])

    print(theta)

    # sampling
    e_xs = []
    e_ys = []

    for t in np.arange(0, 2. * math.pi, 0.01):
        e_xs.append(a * math.cos(t))
        e_ys.append(b * math.sin(t))

    # 回転変換
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

    co : numpy.ndarray

    kai_val : float

    Returns
    ------------



    """
    eig_val, eig_vec = np.linalg.eig(co)

    print("eig_val = {0}".format(eig_val))
    print("eig_vec = {0}".format(eig_vec))

    a = math.sqrt(np.max(eig_val) * kai_val)
    b = math.sqrt(np.min(eig_val) * kai_val)

    eig_num = np.argmax(eig_val)

    theta = math.atan2(eig_vec[eig_num][1], eig_vec[eig_num][0])

    print(theta)

    print("a = {0}".format(a))
    print("b = {0}".format(b))

    e_points = Ellipse(m, 2 * a, 2 * b, angle=-math.degrees(theta))

    return e_points

def main():
    sample_num = 1000
    m = np.array([1., 3.])
    co = np.array([[1., 0.0], [0.0, 1.]])

    ys = np.random.multivariate_normal(m, co, sample_num)

    # true gaussian
    fig = plt.figure()
    
    axis = fig.add_subplot(111)

    axis.set_aspect('equal')

    # points
    axis.scatter(ys[:, 0], ys[:, 1])
    e_xs, e_ys = draw_elipse(m, co)

    e_points = draw_elipse_test(m, co)

    axis.plot(e_xs, e_ys)
    
    axis.add_patch(e_points)
    
    
    plt.show()

if __name__ == "__main__":
    main()
