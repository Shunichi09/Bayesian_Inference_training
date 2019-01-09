import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.font_manager as fon
import sys
import math

# default setting of figures
plt.rcParams["mathtext.fontset"] = 'stix' # math fonts
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
plt.rcParams["font.size"] = 10
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid\

def draw_elipse(m, cov, kai_val=5.991):
    """make elipse with mean and co-varince matrix
    Parameters
    -----------
    m : numpy.ndarray
        mean
    cov : numpy.ndarray
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
    eig_val, eig_vec = np.linalg.eig(cov)

    print("eig_val = {0}".format(eig_val))
    print("eig_vec = {0}".format(eig_vec))

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

class AnimDrawer():
    """create animation of Bayesian inference
    
    Attributes
    ------------
    anim_fig : figure of matplotlib
    axis : axis of matplotlib
    """
    def __init__(self, objects, observation_points=None):
        """
        Parameters
        ------------
        objects : list of objects
        """
        self.observation_points = observation_points
        self.history_means = objects[0]
        self.history_covs = objects[1]
        self.true_mean = objects[2]
        self.true_cov = objects[3]
        self.history_KL_score = objects[4]

        # setting up figure
        self.anim_fig = plt.figure(dpi=150)
        self.axis = self.anim_fig.add_subplot(111)

        # imgs
        self.elipse_imgs = []
        self.point_imgs = []
        self.mean_point_img = []
        self.step_text = None
        
    def draw_anim(self, interval=250):
        """draw the animation and save
        Parameteres
        -------------
        interval : int, optional
            animation's interval time, you should link the sampling time of systems
            default is 50 [ms]
        """
        self._set_axis()
        self._set_img()

        frame_num = len(self.history_means)

        animation = ani.FuncAnimation(self.anim_fig, self._update_anim, interval=interval, frames=frame_num)

        # self.axis.legend()
        print('save_animation?')
        shuold_save_animation = int(input())

        if shuold_save_animation: 
            print('animation_number?')
            num = int(input())
            animation.save('animation_{0}.mp4'.format(num), writer='ffmpeg')
            # animation.save("Sample.gif", writer = 'imagemagick') # gif保存

        plt.show()

    def _set_axis(self):
        """ initialize the animation axies
        """
        # set the axis name
        self.axis.set_xlabel(r'$\it{x}$')
        self.axis.set_ylabel(r'$\it{y}$')
        # self.axis.set_aspect('equal', adjustable='box')

        # set the xlim and ylim        
        self.axis.set_xlim(-5, 5)
        self.axis.set_ylim(-5, 5)         

    def _set_img(self):
        """ initialize the imgs of animation
            this private function execute the make initial imgs for animation
        """
        # text
        self.step_text = self.axis.text(0.05, 0.9, '', transform=self.axis.transAxes)

        
        # observed point
        point_color_list = ["r"]
        for i in range(len(self.history_means)):
            temp_img, = self.axis.plot([],[], ".", color=point_color_list[0], linestyle="dashed")
            self.point_imgs.append(temp_img)

        # object imgs
        # elipse
        line_color_list = ["k", "b"]
        line_style_list = ["solid", "dashed"]

        for i in range(len(line_color_list)):
            temp_img, = self.axis.plot([], [], color=line_color_list[i], linestyle=line_style_list[i])
            self.elipse_imgs.append(temp_img)
        
        # mean points
        point_color_list = ["k", "b"]
        for i in range(len(point_color_list)):
            temp_img, = self.axis.plot([],[], "*", color=point_color_list[i], linestyle="dashed")
            self.mean_point_img.append(temp_img)
    
    def _update_anim(self, i):
        """the update animation
        this function should be used in the animation functions
        Parameters
        ------------
        i : int
            time step of the animation
            the sampling time should be related to the sampling time of system
        Returns
        -----------
        object_imgs : list of img
        traj_imgs : list of img
        """
        if self.observation_points is not None:
            self.step_text.set_text('points_num = {0}, KL = {1}'.format(i + 5, self.history_KL_score[i]))
        else:
            self.step_text.set_text('KL = {0}'.format(self.history_KL_score[i]))

        self._draw_objs(i)

        return self.elipse_imgs, self.mean_point_img, self.point_imgs
        
    def _draw_objs(self, i):
        """
        This private function is just divided thing of
        the _update_anim to see the code more clear
        Parameters
        ------------
        i : int
            time step of the animation
            the sampling time should be related to the sampling time of system
        """
        # make elipse
        # true
        e_xs, e_ys = draw_elipse(self.true_mean, self.true_cov)

        # esti
        est_e_xs, est_e_ys = draw_elipse(self.history_means[i], self.history_covs[i])

        # plot
        # true
        self.elipse_imgs[0].set_data(e_xs, e_ys)
        # est
        self.elipse_imgs[1].set_data(est_e_xs, est_e_ys)

        # obse point
        if self.observation_points is not None:
            for j in range(i + 1):
                self.point_imgs[j].set_data(self.observation_points[0, j], self.observation_points[1, j])

        # mean points
        # true
        self.mean_point_img[0].set_data(self.true_mean[0], self.true_mean[1])
        # estimate
        self.mean_point_img[1].set_data(self.history_means[i][0], self.history_means[i][1])