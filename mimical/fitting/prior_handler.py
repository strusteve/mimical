from nautilus import Prior
from scipy.stats import norm
import numpy as np

class priorHandler(object):
    """ Contains functions for translating Mimical user priors into 
    Nautlius priors and translating sampler samples into model
    paremeters for each filter.

    Parameters
    ----------

    user_prior : dict
        The user specified prior which set out the priors for the model parameters
        and passes information about whether to let these vary for each filter or
        whether they follow an order-specified polynomial relationship.

    filter_names : list - str
        A list of filter names e.g., [F356W, F444W, ...]

    wavs : array
        A 1D array of effective wavelengths corresponding to each filter.
    
    """


    def __init__(self, user_prior, filter_names, wavs):
        self.user_prior = user_prior
        self.filter_names = filter_names
        self.wavs = wavs

        self.nmodel, self.nparam, self.ndim = self.calculate_dimensionality()

    
    def calculate_dimensionality(self):
        """ Calculates the model parameters, Mimical parameters and dimensionality of the sampling algorithm. """

        nmodel = 0
        nparam = 0
        ndim = 0

        # Loop over model parameters
        for key in self.user_prior.keys():

            if ('rms' not in key) & ('flux_to_counts' not in key):
                nmodel += 1

            param_prior_traits = self.user_prior[key]
            param_prior_dist = param_prior_traits[0]
            param_fit_type = param_prior_traits[1]
            
            # For fitted params
            if type(param_prior_dist).__name__ == 'tuple':

                if param_fit_type == "Individual":
                    for i in range(len(self.wavs)):
                        nparam+=1
                        ndim+=1       

                elif param_fit_type == "Polynomial":
                    poly_order = param_prior_traits[2]
                    for i in range(0,poly_order+1):
                        nparam+=1
                        ndim+=1

            # For fixed params
            elif (type(param_prior_dist).__name__ == 'float') | (type(param_prior_dist).__name__ == 'int') | (type(param_prior_dist).__name__ == 'list') | (type(param_prior_dist).__name__ == 'ndarray'):
            
                if param_fit_type == "Individual":
                    for i in range(len(self.wavs)):
                        nparam+=1

                elif param_fit_type == "Polynomial":
                    poly_order = param_prior_traits[2]
                    for i in range(0,poly_order+1):
                        nparam+=1

        return nmodel,nparam, ndim
    

    def generate_sampler_prior_keys(self):
        """ Generates the keys or labels for the sampled parameters. """

        keys = []

        # Loop over model parameters
        for key in self.user_prior.keys():
            param_prior_traits = self.user_prior[key]
            param_fit_type = param_prior_traits[1]

            # If user specifies 'Individual', add a fitter free-parameter for each filter.
            if param_fit_type == "Individual":
                for i in range(len(self.wavs)):
                    keys.append(f'{key}_{self.filter_names[i]}')
            
            # If user specifies 'Polynomial', add a fitter free-parameter for each coefficient. 
            # e.g., For order 0, only one free-parameter is included for the whole fitting run a.k.a constant between filters.
            # e.g., For order 1, two free-parameter are included for the whole fitting run a.k.a straight-line relationship  with effective wavelength.
            # The lowest wavelength is chosen as the origin.
            elif param_fit_type == "Polynomial":
                poly_order = param_prior_traits[2]
                for i in range(0,poly_order+1):
                    keys.append(key+f'_C{i}')

            else:
                raise Exception("Fitting type not supported, please choose either 'Individual' or 'Polynomial'.")
        
        return keys


    def sampler_prior(self, x):
        """ Defines the prior used for sampling. Transforms the unit cube. """

        # Unit cube
        theta = np.copy(x)
        count = 0
        
        # Loop over model parameters
        for key in self.user_prior.keys():

            param_prior_traits = self.user_prior[key]
            param_prior_dist = param_prior_traits[0]
            param_fit_type = param_prior_traits[1]

            # For fitted params
            if type(param_prior_dist).__name__ == 'tuple':

                # If user specifies 'Individual', add a fitter free-parameter for each filter.
                if param_fit_type == "Individual":
                    for i in range(len(self.wavs)):
                        theta[count] = (theta[count] * (param_prior_dist[1]-param_prior_dist[0])) + param_prior_dist[0]
                        count+=1
                
                # If user specifies 'Polynomial', add a fitter free-parameter for each coefficient. 
                # e.g., For order 0, only one free-parameter is included for the whole fitting run a.k.a constant between filters.
                # e.g., For order 1, two free-parameter are included for the whole fitting run a.k.a straight-line relationship  with effective wavelength.
                # The lowest wavelength is chosen as the origin.
                elif param_fit_type == "Polynomial":

                    theta[count] = (theta[count] * (param_prior_dist[1]-param_prior_dist[0])) + param_prior_dist[0]
                    countinit = count
                    count+=1

                    poly_order = param_prior_traits[2]
                    random_order = np.append(0, np.random.choice(np.arange(poly_order), size=poly_order, replace=False)+1)

                    # Calculate the conditional priors for higher order polynomial coefficients.
                    for i in range(1, len(random_order)):

                        prev_coeffs = theta[(countinit+random_order)[:i]]
                        prev_polywavs = np.power(self.wavs[-1]-self.wavs[0], random_order[:i])
                        prev_comps = prev_coeffs * prev_polywavs
                        prev_comps_summed = np.sum(prev_comps)
                            
                        min = (param_prior_dist[0] - prev_comps_summed) / (np.power(self.wavs[-1]-self.wavs[0], random_order[i]))
                        max = (param_prior_dist[1] - prev_comps_summed) / (np.power(self.wavs[-1]-self.wavs[0], random_order[i]))
                        
                        theta[countinit+random_order[i]] = (theta[countinit+random_order[i]] * (max-min)) + min
                        count+=1
                    
                else:
                    raise Exception("Fitting type not supported, please choose either 'Individual' or 'Polynomial'.")

            # For fixed params
            elif (type(param_prior_dist).__name__ == 'float') | (type(param_prior_dist).__name__ == 'int') | (type(param_prior_dist).__name__ == 'list') | (type(param_prior_dist).__name__ == 'ndarray'):
                
                # If fixed for each individual filter, set for each separately
                if param_fit_type == "Individual":
                    
                    # Helper for single image fit
                    if len(self.wavs) == 1:
                        if not type(param_prior_dist).__name__ == 'ndarray':
                            theta[count] = param_prior_dist
                            count+=1
                        else:
                            theta[count] = np.mean(param_prior_dist)
                            count+=1
                            
                    # For multiple image fits
                    else:
                        for i in range(len(self.wavs)):
                            # If use supplies single value for each filter, simply set.
                            if not (type(param_prior_dist[i]).__name__ == 'ndarray'):
                                theta[count] = param_prior_dist[i]
                                count+=1
                            # If user supplies values for each image pixel (pertinent for RMS etc.), then pass the
                            # mean to the prior samples. This is required for generality but is overwritten later in the 
                            # likelihood function.
                            else:
                                theta[count] = np.mean(param_prior_dist[i])
                                count+=1

                # If user supplies polynomial coefficients, set them.
                elif param_fit_type == "Polynomial":
                    poly_order = param_prior_traits[2]
                    if poly_order==0:
                        if (type(param_prior_dist).__name__ == 'float') | (type(param_prior_dist).__name__ == 'int'):
                            theta[count] = param_prior_dist
                            count+=1
                        elif (type(param_prior_dist).__name__ == 'list') | (type(param_prior_dist).__name__ == 'ndarray'):
                            theta[count] = param_prior_dist[0]
                            count+=1
                    else:
                        for i in range(0,poly_order+1):
                            theta[count] = param_prior_dist[i] 
                            count+=1

                else:
                    raise Exception("Fitting type not supported, please choose either 'Individual' or 'Polynomial'.")
            
            else:
                raise Exception("Mimical only accepts a min/max tuple for fitting, or a list/ndarray/float/int for fixing.")

        return theta


    def translate(self):
        """ Translate a Mimical prior into a sampler prior."""
        return self.sampler_prior


    def revert(self, param_dict):
        """ Translate a sampler sample into a sample of model parameters for each filter."""


        # Empty parameter array
        params_final = np.zeros((len(self.wavs), len(self.user_prior.keys())))

        # Loop over model parameters
        keys = list(self.user_prior.keys())

        count = 0
        for i in range(len(keys)):
            param_prior_traits = self.user_prior[keys[i]]
            param_fit_type = param_prior_traits[1]

            # If individual, add the sampler sample for each filter
            if param_fit_type == "Individual":
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

        return params_final
    

    def check_priors_sampler(self, n):

        unit_cube = np.random.rand(n, self.nparam)
        samples_mimical = np.apply_along_axis(self.sampler_prior, 1, unit_cube)
        
        return samples_mimical, self.generate_sampler_prior_keys()
    

    def check_priors_mimical(self, n):

        unit_cube = np.random.rand(n, self.nparam)
        samples_mimical = np.apply_along_axis(lambda unit_vec: self.revert(self.sampler_prior(unit_vec)).flatten(), 1, unit_cube)
        
        # Save to .csv table
        keys = []
        for j in range(len(self.filter_names)):
            for i in range(len(self.user_prior.keys())):
                key = list(self.user_prior.keys())[i]
                keys.append(f"{key}_{self.filter_names[j]}")


        return samples_mimical, keys





