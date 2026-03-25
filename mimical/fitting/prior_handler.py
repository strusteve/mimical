from nautilus import Prior
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits

from .prior_types import individual, polynomial, powerlaw

dir_path = os.getcwd()


class priorHandler(object):
    """ Contains the functionality for translating Mimical priors into 
    sampler priors, and translating sampler samples into model
    parameters in each filter.

    Parameters
    ----------

    mimical_prior : dict
        The user specified prior which set out the priors for the model parameters
        and passes information about whether to let these vary for each filter or
        whether they follow an order-specified polynomial relationship.

    filter_names : list - str
        A list of filter names e.g., [F356W, F444W, ...]

    wavs : array
        A 1D array of effective wavelengths corresponding to each filter.
    
    """



    def __init__(self, mimical_prior, filter_names, wavs, images, runtag, id):
        self.mimical_prior = mimical_prior
        self.filter_names = filter_names
        self.wavs = wavs
        self.nmodel, self.nparam, self.ndim = self.calculate_dimensionality()
        self.samplemask = self.calculate_sampler_mask()
        self.images = images
        self.runtag = runtag
        self.id = id


    def sampler_prior(self, x):
        """ Defines the prior used for sampling. Transforms the unit cube. """

        # Create empty parameter array
        theta = np.zeros(self.nparam)

        # Keep record of the current element in the unit cube
        xcount = 0
        # Keep record of the current element in the parameter array
        thetacount = 0
        
        # Loop over Mimical parameters
        for key in self.mimical_prior.keys():
            
            # Load the current Mimical element
            param_prior_traits = self.mimical_prior[key]
            param_prior_dist = param_prior_traits[0]

            # For fitted params
            if type(param_prior_dist).__name__ == 'tuple':

                param_fit_type = param_prior_traits[1]

                # If user specifies 'Individual', add a free parameter for each filter.
                if param_fit_type == "Individual":
                    theta[thetacount:thetacount+len(self.wavs)] = individual(x[xcount:xcount+len(self.wavs)], param_prior_dist)
                    thetacount+=len(self.wavs)
                    xcount+=len(self.wavs)

                # If user specifies 'Polynomial', add a free parameter for each polynomial coefficient. 
                elif param_fit_type == "Polynomial":
                    poly_order = param_prior_traits[2]
                    theta[thetacount:thetacount+poly_order+1] = polynomial(x[xcount:xcount+poly_order+1], param_prior_dist, poly_order, self.wavs)
                    thetacount+=poly_order+1
                    xcount+=poly_order+1

                # If user specifies 'Power-law', add a free parameter for each power law coefficient. 
                elif param_fit_type == "Power-law":
                    powerbounds = param_prior_traits[2]
                    epsilon = param_prior_traits[3]
                    theta[thetacount:thetacount+3] = powerlaw(x[xcount:xcount+3], param_prior_dist, self.wavs, powerbounds, epsilon)
                    thetacount+=3
                    xcount+=3

                else:
                    raise Exception("Fitting type not supported, please choose either 'Individual', 'Polynomial' or 'Power-law'.")

            # For fixed params
            elif (type(param_prior_dist).__name__ == 'float') | (type(param_prior_dist).__name__ == 'int') | (type(param_prior_dist).__name__ == 'list') | (type(param_prior_dist).__name__ == 'ndarray'):
                
                param_fit_type = param_prior_traits[1]

                # If fixed for each individual filter, set for each separately
                if (param_fit_type == "Individual"):
                    
                    # Helper for single image fit
                    if len(self.wavs) == 1:
                        if not type(param_prior_dist).__name__ == 'ndarray':
                            theta[thetacount] = param_prior_dist
                            thetacount+=1
                        else:
                            theta[thetacount] = np.mean(param_prior_dist)
                            thetacount+=1
                            
                    # For multiple image fits
                    else:
                        if (type(param_prior_dist).__name__ == 'float') | (type(param_prior_dist).__name__ == 'int'):
                            for i in range(len(self.wavs)):
                                theta[thetacount] = param_prior_dist
                                thetacount+=1

                        elif (type(param_prior_dist).__name__ == 'list'):
                            for i in range(len(self.wavs)):
                                # If use supplies single value for each filter, simply set.
                                if not (type(param_prior_dist[i]).__name__ == 'ndarray'):
                                    theta[thetacount] = param_prior_dist[i]
                                    thetacount+=1
                                # If user supplies values for each image pixel (pertinent for RMS etc.), then pass the
                                # mean to the prior samples. This is required for generality but is overwritten later in the 
                                # likelihood function.
                                else:
                                    theta[thetacount] = np.mean(param_prior_dist[i])
                                    thetacount+=1
                            
                        else: 
                            raise Exception('Must pass float/int/list for a multiband fit. The list can be a list of floats/ints or a list of arrays.')


                # If user supplies polynomial coefficients, set them.
                elif param_fit_type == "Polynomial":
                    poly_order = param_prior_traits[2]
                    if poly_order==0:
                        if (type(param_prior_dist).__name__ == 'float') | (type(param_prior_dist).__name__ == 'int'):
                            theta[thetacount] = param_prior_dist
                            thetacount+=1
                        elif (type(param_prior_dist).__name__ == 'list') | (type(param_prior_dist).__name__ == 'ndarray'):
                            theta[thetacount] = param_prior_dist[0]
                            thetacount+=1
                    else:
                        for i in range(0, poly_order+1):
                            theta[thetacount] = param_prior_dist[i] 
                            thetacount+=1
                
                # If user supplies power-law coefficients, set them.
                elif param_fit_type == "Power-law":
                        for i in range(0, 3):
                            theta[thetacount] = param_prior_dist[i] 
                            thetacount+=1

                else:
                    raise Exception("Fitting type not supported, please choose either 'Individual', 'Polynomial' or 'Power-law'.")
            
            # For inferred RMS
            elif (key=='rms') & (type(param_prior_dist).__name__=='str'):

                if (param_prior_dist=='Infer'):
                    
                    for i in range(len(self.wavs)):
                        if os.path.isfile(dir_path+f'/mimical/sextractor/segmaps{self.runtag}' + f'/{self.id}_{self.filter_names[i]}.fits'):
                                segmap = fits.open(dir_path+f'/mimical/sextractor/segmaps{self.runtag}' + f'/{self.id}_{self.filter_names[i]}.fits')[0].data
                                bckgnd = self.images[i][segmap==0]
                                rms = ((np.sum(bckgnd**2))/len(bckgnd))**(1/2)
                                theta[thetacount] = rms
                                thetacount+=1
                        else:
                            raise Exception('')
                    
                else: raise Exception('')

            else:
                raise Exception("Mimical only accepts a min/max tuple for fitting, or a list/ndarray/float/int for fixing.")

        return theta
    

    def revert(self, param_dict):
        """ Translate a sampler sample into a sample of model parameters for each filter."""


        # Empty parameter array
        params_final = np.zeros((len(self.wavs), len(self.mimical_prior.keys())))

        # Loop over model parameters
        keys = list(self.mimical_prior.keys())
        count = 0

        for i in range(len(keys)):
            param_prior_traits = self.mimical_prior[keys[i]]
            param_prior_dist = param_prior_traits[0]

            if (keys[i]=='rms') & (type(param_prior_dist).__name__=='str'):
                if  (param_prior_dist=='Infer'):
                    for j in range(len(self.wavs)):
                        params_final[j,i] = param_dict[count]
                        count+=1
                else: raise Exception(' ')

            else:

                param_fit_type = param_prior_traits[1]

                # If individual, add the sample for each filter
                if (param_fit_type == "Individual"):
                    for j in range(len(self.wavs)):
                        params_final[j,i] = param_dict[count]
                        count+=1
                
                # If polynomial, calculate the expected parameter in each filter given its effective wavlength
                elif param_fit_type == "Polynomial":
                    poly_order = param_prior_traits[2]
                    coeffs = param_dict[count:count+poly_order+1]
                    polywavs = np.power(np.tile(self.wavs-self.wavs[0], (poly_order+1,1)).T, np.arange(poly_order+1))
                    comps = coeffs * polywavs
                    comps_summed = np.sum(comps, axis=1)
                    params_final[:,i] = comps_summed       
                    count+=poly_order+1 
                
                # If power-law, calculate the expected parameter in each filter given its effective wavlength
                elif param_fit_type == "Power-law":
                    epsilon = param_prior_traits[3]
                    coeffs = param_dict[count:count+3]
                    polywavs = np.power(np.tile(((self.wavs-self.wavs[0])+epsilon)/((self.wavs[-1]-self.wavs[0])+epsilon), (2,1)).T, [0,coeffs[2]])
                    comps = np.array(([coeffs[0], coeffs[1]-coeffs[0]])) * polywavs
                    comps_summed = np.sum(comps, axis=1)
                    params_final[:,i] = comps_summed  
                    count+=3
                
                else:
                    raise Exception("Fitting type not supported, please choose either 'Individual', 'Polynomial' or 'Power-law'.")

        return params_final
    


#################################################################################################################



    def calculate_dimensionality(self):
        """ Calculates the model parameters, Mimical parameters and dimensionality of the sampling algorithm. """

        nmodel = 0
        nparam = 0
        ndim = 0

        # Loop over model parameters
        for key in self.mimical_prior.keys():

            # If Mimical paramter is not part of the noise, add it to the model parameter count
            if ('rms' not in key) & ('counts_per_flux' not in key):
                nmodel += 1

            # Load in the Mimical prior element
            param_prior_traits = self.mimical_prior[key]
            param_prior_dist = param_prior_traits[0]
            
            # For fitted params
            if type(param_prior_dist).__name__ == 'tuple':

                param_fit_type = param_prior_traits[1]

                if param_fit_type == "Individual":
                    for i in range(len(self.wavs)):
                        nparam+=1
                        ndim+=1       

                elif param_fit_type == "Polynomial":
                    poly_order = param_prior_traits[2]
                    for i in range(0,poly_order+1):
                        nparam+=1
                        ndim+=1
                
                elif param_fit_type == "Power-law":
                    nparam+=3
                    ndim+=3

                else:
                    raise Exception("Fitting type not supported, please choose either 'Individual', 'Polynomial' or 'Power-law'.")

            # For fixed params
            elif (type(param_prior_dist).__name__ == 'float') | (type(param_prior_dist).__name__ == 'int') | (type(param_prior_dist).__name__ == 'list') | (type(param_prior_dist).__name__ == 'ndarray'):
                
                param_fit_type = param_prior_traits[1]

                if param_fit_type == "Individual":
                    for i in range(len(self.wavs)):
                        nparam+=1

                elif param_fit_type == "Polynomial":
                    poly_order = param_prior_traits[2]
                    for i in range(0,poly_order+1):
                        nparam+=1

                elif param_fit_type == "Power-law":
                    nparam+=3
                
                else:
                    raise Exception("Fitting type not supported, please choose either 'Individual', 'Polynomial' or 'Power-law'.")
            
            # For inferred RMS
            elif (key=='rms') & (type(param_prior_dist).__name__=='str'):
                if (param_prior_dist=='Infer'):
                    for i in range(len(self.wavs)):
                        nparam+=1
                else: raise Exception('')


        return nmodel,nparam,ndim
    


    def calculate_sampler_mask(self):
        """ Generates a mask over the Mimical samples which were not fixed in the sampling step. """

        nparam = 0
        samplemask = np.zeros(self.nparam) == np.zeros(self.nparam)

        # Loop over model parameters
        for key in self.mimical_prior.keys():

            # Load in the Mimical prior element
            param_prior_traits = self.mimical_prior[key]
            param_prior_dist = param_prior_traits[0]
            
            # For fitted params
            if type(param_prior_dist).__name__ == 'tuple':

                param_fit_type = param_prior_traits[1]

                if param_fit_type == "Individual":
                    for i in range(len(self.wavs)):
                        nparam+=1    

                elif param_fit_type == "Polynomial":
                    poly_order = param_prior_traits[2]
                    for i in range(0,poly_order+1):
                        nparam+=1
            

                elif param_fit_type == "Power-law":
                        nparam+=3
                
                else:
                    raise Exception('')

            # For fixed params
            elif (type(param_prior_dist).__name__ == 'float') | (type(param_prior_dist).__name__ == 'int') | (type(param_prior_dist).__name__ == 'list') | (type(param_prior_dist).__name__ == 'ndarray'):
                
                param_fit_type = param_prior_traits[1]

                if param_fit_type == "Individual":
                    for i in range(len(self.wavs)):
                        samplemask[nparam]=False
                        nparam+=1

                elif param_fit_type == "Polynomial":
                    poly_order = param_prior_traits[2]
                    for i in range(0,poly_order+1):
                        samplemask[nparam]=False
                        nparam+=1

                elif param_fit_type == "Power-law":
                    samplemask[nparam:nparam+3]=False
                    nparam+=3

                else:
                    raise Exception("Fitting type not supported, please choose either 'Individual', 'Polynomial' or 'Power-law'.")
            
            # For inferred RMS
            elif (key=='rms') & (type(param_prior_dist).__name__=='str'):
                if (param_prior_dist=='Infer'):
                    for i in range(len(self.wavs)):
                        nparam+=1
                else: raise Exception('')

        return samplemask
    


    def generate_sampler_prior_keys(self):
        """ Generates the keys or labels for the sampled parameters. """

        keys = []

        # Loop over model parameters
        for key in self.mimical_prior.keys():
            param_prior_traits = self.mimical_prior[key]
            param_prior_dist = param_prior_traits[0]


            if (key=='rms') & (type(param_prior_dist).__name__=='str'):
                if (param_prior_dist=='Infer'):
                    for i in range(len(self.wavs)):
                        keys.append(f'{key}_{self.filter_names[i]}')
                else: raise Exception('')

            else:
                param_fit_type = param_prior_traits[1]

                # Add parameters for each filter
                if param_fit_type == "Individual":
                    for i in range(len(self.wavs)):
                        keys.append(f'{key}_{self.filter_names[i]}')
                
                # Add parameters for each polynomial coefficient
                elif param_fit_type == "Polynomial":
                    poly_order = param_prior_traits[2]
                    for i in range(0,poly_order+1):
                        keys.append(key+f'_C{i}')
                
                elif param_fit_type == "Power-law":
                    for i in range(0,3):
                        keys.append(key+f'_P{i}')

                else:
                    raise Exception("Fitting type not supported, please choose either 'Individual', 'Polynomial' or 'Power-law'.")
            
        return keys

   

##############################################################################################################################



    def check_priors_sampler(self, n):
        """ Sample the fitted prior volume n times."""

        unit_cube = np.random.rand(n, self.nparam)
        samples_mimical = np.apply_along_axis(self.sampler_prior, 1, unit_cube)
        
        return samples_mimical, self.generate_sampler_prior_keys()
    


    def check_priors_mimical(self, n):
        """ Sample the Mimical prior volume n times."""

        unit_cube = np.random.rand(n, self.nparam)
        samples_mimical = np.apply_along_axis(lambda unit_vec: self.revert(self.sampler_prior(unit_vec)).flatten(), 1, unit_cube)
        
        keys = []
        for j in range(len(self.filter_names)):
            for i in range(len(self.mimical_prior.keys())):
                key = list(self.mimical_prior.keys())[i]
                keys.append(f"{key}_{self.filter_names[j]}")

        return samples_mimical, keys
    
    
    
    def check_physical(self, n):
        """ Check what fraction of Mimical prior samples are physical under the user constraints. """

        unit_cube = np.random.rand(n, self.nparam)
        samples_mimical = np.apply_along_axis(lambda unit_vec: self.revert(self.sampler_prior(unit_vec)).flatten(), 1, unit_cube)
        
        mask = np.arange(n) == np.arange(n)

        for i in range(n):
            samples_now = samples_mimical[i].reshape(len(self.wavs), len(list(self.mimical_prior.keys())))
            for j in range(len(self.mimical_prior.keys())):
                if (samples_now[:,j]<self.mimical_prior[list(self.mimical_prior.keys())[j]][0][0]).any() | (samples_now[:,j]>self.mimical_prior[list(self.mimical_prior.keys())[j]][0][1]).any():
                    mask[i]=False
                else:
                    continue
            
        return np.sum(mask)/len(mask)
    

    def plot_samples(self, n, key):
        """ Check what fraction of Mimical prior samples are physical under the user constraints. """

        unit_cube = np.random.rand(n, self.nparam)
        samples_mimical = np.apply_along_axis(lambda unit_vec: self.revert(self.sampler_prior(unit_vec)).flatten(), 1, unit_cube)

        mask = np.arange(n) == np.arange(n)

        for i in range(n):
            samples_now = samples_mimical[i].reshape(len(self.wavs), len(list(self.mimical_prior.keys())))
            for j in range(len(self.mimical_prior.keys())):
                if self.mimical_prior[list(self.mimical_prior.keys())[j]][0] == 'tuple':
                    if (samples_now[:,j]<self.mimical_prior[list(self.mimical_prior.keys())[j]][0][0]).any() | (samples_now[:,j]>self.mimical_prior[list(self.mimical_prior.keys())[j]][0][1]).any():
                        mask[i]=False
                    else: continue
                else: continue

        physical = samples_mimical#[mask]
        fig, ax = plt.subplots()
        for i in range(len(physical)):
            curr_sample = physical[i].reshape(len(self.wavs), len(list(self.mimical_prior.keys())))
            ofinterest = curr_sample[:, list(self.mimical_prior.keys()).index(key)]
            ax.plot(self.wavs, ofinterest, color='black', alpha=1)
        
        ax.set_ylabel(key)
        ax.set_xlabel('$\lambda$')
        plt.show()
            









