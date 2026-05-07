import numpy as np
from scipy.special import gammaincinv

class Sersic(object):
    ''' Basic Sersic Model'''

    def __init__(self, parameters=[0]*7):
        self.param_names = ['amplitude', 'r_eff', 'n', 'x_0', 'y_0', 'ellip', 'theta']
        self.amplitude, self.r_eff, self.n, self.x_0, self.y_0, self.ellip, self.theta = parameters

    def update_parameters(self, parameters):
        self.amplitude, self.r_eff, self.n, self.x_0, self.y_0, self.ellip, self.theta = parameters

    def evaluate(self, x, y):

        bn = gammaincinv(2.0 * self.n, 0.5)
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        x_maj = np.abs((x - self.x_0) * cos_theta + (y - self.y_0) * sin_theta)
        x_min = np.abs(-(x - self.x_0) * sin_theta + (y - self.y_0) * cos_theta)

        b = (1 - self.ellip) * self.r_eff
        expon = 2.0 #+ self.c
        inv_expon = 1.0 / expon
        z = ((x_maj / self.r_eff) ** expon + (x_min / b) ** expon) ** inv_expon

        return self.amplitude * np.exp(-bn * (z ** (1 / self.n) - 1.0))