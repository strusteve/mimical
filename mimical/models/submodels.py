import torch
torch.set_default_dtype(torch.float32)



class Sersic(object):
    """ Basic Sersic submodel, fully vectorised for coordinate cubes with unique parameters for each slice. """



    def __init__(self, parameters=torch.zeros((2,7))):
        
        self.param_names = ['flux', 'r_eff', 'n', 'x_0', 'y_0', 'ellip', 'theta']

        self.params = parameters
        self.flux = parameters[:,0]
        self.r_eff = parameters[:,1]
        self.n = parameters[:,2]
        self.x_0 = parameters[:,3]
        self.y_0 = parameters[:,4]
        self.ellip = parameters[:,5]
        self.theta = parameters[:,6]



    def update_parameters(self, parameters):

        self.params = parameters
        self.flux = parameters[:,0]
        self.r_eff = parameters[:,1]
        self.n = parameters[:,2]
        self.x_0 = parameters[:,3]
        self.y_0 = parameters[:,4]
        self.ellip = parameters[:,5]
        self.theta = parameters[:,6]



    def evaluate(self, x, y):

        # bn approximation from Asali et al. 2025
        a0 = 1.8073182821237496e-4
        a1 = 3.7026973571904255e-5
        a2 = -0.09149183119702775
        a3 = 2.6248718397705195
        a4 = -0.9727511612512357
        a5 = 94.78011643586419
        a6 = -0.006044236674273689
        An = (2*self.n) - (1/3) + (4 / (405 * self.n)) + (46 / (25515 * (self.n**2)))
        Cn = (((a0/(self.n+torch.exp(self.n)))+a1) / ((a2-self.n)**2)) * (torch.log(self.n) + (((a3*self.n)+a4) / ((self.n**(-4)) +a5) / (self.n+a6)))
        bn = An + Cn

        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)

        xT = x.permute(2,1,0)
        yT = y.permute(2,1,0)

        x_maj = (xT - self.x_0) * cos_theta + (yT - self.y_0) * sin_theta
        x_min = -(xT - self.x_0) * sin_theta + (yT - self.y_0) * cos_theta

        b = (1 - self.ellip) * self.r_eff
        z = torch.sqrt((x_maj / self.r_eff) ** 2 + (x_min / b) ** 2)

        # With thanks to Peng et al. 2002, Equation 7
        amplitude = self.flux / (2 * torch.pi * (self.r_eff**2) * torch.exp(bn + torch.special.gammaln(2*self.n)) * self.n * (bn**(-2*self.n)) * (1.-self.ellip))

        final = (amplitude * torch.exp(-bn * (z ** (1 / self.n) - 1.0)))

        return final.permute(2,1,0)