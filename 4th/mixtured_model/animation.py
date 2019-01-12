import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.font_manager as fon
import sys
import math

# default setting of figures
plt.rcParams["mathtext.fontset"] = 'stix' # math fonts
plt.rcParams['axes.grid'] = True # make grid\

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
        self.history_s_sample = objects[0]

        # setting up figure
        self.anim_fig = plt.figure(dpi=150)
        self.axis = self.anim_fig.add_subplot(111)

        # imgs
        self.point_imgs_class_1 = []
        self.point_imgs_class_2 = []
        self.point_imgs_class_3 = []

        self.step_text = None
        
    def draw_anim(self, interval=300):
        """draw the animation and save
        Parameteres
        -------------
        interval : int, optional
            animation's interval time, you should link the sampling time of systems
            default is 50 [ms]
        """
        self._set_axis()
        self._set_img()

        frame_num = len(self.history_s_sample)

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
        margin = 5
        self.axis.set_xlim(np.min(self.observation_points) - margin, np.max(self.observation_points) + margin)
        self.axis.set_ylim(np.min(self.observation_points) - margin, np.max(self.observation_points) + margin)         

    def _set_img(self):
        """ initialize the imgs of animation
            this private function execute the make initial imgs for animation
        """
        # text
        self.step_text = self.axis.text(0.05, 0.9, '', transform=self.axis.transAxes)

        # observed point
        point_color_list = ["r", "b", "g"]
        for i in range(len(self.observation_points)):
            temp_img, = self.axis.plot([],[], ".", color=point_color_list[0])
            self.point_imgs_class_1.append(temp_img)

        for i in range(len(self.observation_points)):
            temp_img, = self.axis.plot([],[], ".", color=point_color_list[1])
            self.point_imgs_class_2.append(temp_img)

        for i in range(len(self.observation_points)):
            temp_img, = self.axis.plot([],[], ".", color=point_color_list[2])
            self.point_imgs_class_3.append(temp_img)
    
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
        self.step_text.set_text('iteration =  {0}'.format(i))
        self._draw_objs(i)

        return self.point_imgs_class_1, self.point_imgs_class_2
        
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
        
        for j, datum in enumerate(self.observation_points):

            self.point_imgs_class_1[j].set_data([], [])
            self.point_imgs_class_2[j].set_data([], [])
            self.point_imgs_class_3[j].set_data([], [])

            if self.history_s_sample[i][j]  == 0:
                self.point_imgs_class_1[j].set_data(datum[0], datum[1])
            elif self.history_s_sample[i][j] == 1:
                self.point_imgs_class_2[j].set_data(datum[0], datum[1])
            elif self.history_s_sample[i][j] == 2: 
                self.point_imgs_class_3[j].set_data(datum[0], datum[1])