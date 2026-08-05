import torch
torch.set_default_dtype(torch.float32)


class Sersic(object):
    """ Basic Sersic submodel, fully vectorised for coordinate cubes. """

    def __init__(self, parameters=torch.zeros((2, 7)), zp=23.9):

        self.param_names = ['mag', 'r_eff', 'n', 'x_0',
                            'y_0', 'ellip', 'theta']
        self.zp = zp
        self.update_parameters(parameters)

    def update_parameters(self, parameters):

        self.params = parameters
        self.mag = parameters[:, 0]
        self.r_eff = parameters[:, 1]
        self.n = parameters[:, 2]
        self.x_0 = parameters[:, 3]
        self.y_0 = parameters[:, 4]
        self.ellip = parameters[:, 5]
        self.theta = parameters[:, 6]

        self.inv_r_eff = 1 / self.r_eff
        self.inv_n = 1 / self.n

        # bn approximation from Asali et al. 2025
        a0 = 1.8073182821237496e-4
        a1 = 3.7026973571904255e-5
        a2 = -0.09149183119702775
        a3 = 2.6248718397705195
        a4 = -0.9727511612512357
        a5 = 94.78011643586419
        a6 = -0.006044236674273689
        An = ((2*self.n) - (1/3) + (4 / (405 * self.n)) +
              (46 / (25515 * (self.n.square()))))
        Cn = ((((a0/(self.n+torch.exp(self.n))) + a1) / ((a2-self.n).square()))
              * (torch.log(self.n) + (((a3*self.n) + a4) /
                                      ((self.n**(-4)) + a5) /
                                      (self.n + a6))))
        self.bn = An + Cn

        self.cos_theta = torch.cos(self.theta)
        self.sin_theta = torch.sin(self.theta)
        self.b = (1 - self.ellip) * self.r_eff
        self.inv_b = 1 / self.b

        flux_dens = torch.pow(10, (self.zp-self.mag)/2.5)
        # With thanks to Peng et al. 2002, Equation 7
        self.amplitude = flux_dens / (2 * torch.pi * (self.r_eff.square()) *
                                      torch.exp(self.bn +
                                                torch.special.gammaln(2 *
                                                                      self.n))
                                      * self.n * (self.bn**(-2*self.n)) *
                                      (1.-self.ellip))

    def evaluate(self, x, y):

        dx = x - self.x_0[:, None, None]
        dy = y - self.y_0[:, None, None]

        cos = self.cos_theta[:, None, None]
        sin = self.sin_theta[:, None, None]

        x_maj = dx * cos + dy * sin
        x_min = -dx * sin + dy * cos

        r2 = (x_maj * self.inv_r_eff[:, None, None]).square()
        r2 += (x_min * self.inv_b[:, None, None]).square()

        final = (self.amplitude[:, None, None] *
                 torch.exp(-self.bn[:, None, None] *
                           (torch.pow(r2, 0.5 *
                                      self.inv_n[:, None, None]) - 1)))

        return final


class Point(object):
    """ Basic point-source (pixel-size) submodel, fully vectorised for coordinate cubes. """

    def __init__(self, parameters=torch.zeros((2, 3)), zp=23.9):

        self.param_names = ['mag', 'x_0', 'y_0']
        self.zp = zp
        self.update_parameters(parameters)

    def update_parameters(self, parameters):

        self.params = parameters
        self.mag = parameters[:, 0]
        self.x_0 = parameters[:, 1]
        self.y_0 = parameters[:, 2]

        self.flux_dens = torch.pow(10, (self.zp-self.mag)/2.5)

    def evaluate(self, x, y):

        dx = (x - self.x_0[:, None, None]).square()
        dy = (y - self.y_0[:, None, None]).square()

        final = ((dx < 0.25) & (dy < 0.25))*self.flux_dens

        return final
