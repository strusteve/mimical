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

    
    def calculate_dimensionality(self):
        """ Calculates the dimensionality of the sampling algorithm. """

        ndim = 0
        # Loop over model parameters
        for key in self.user_prior.keys():
            param_prior_traits = self.user_prior[key]
            param_fit_type = param_prior_traits[1]

            if param_fit_type == "Individual":
                for i in range(len(self.wavs)):
                    ndim+=1
            
            elif param_fit_type == "Polynomial":
                ndim+=1
                poly_order = param_prior_traits[2]
                for i in range(1,poly_order+1):
                    ndim+=1

            else:
                raise Exception("Fitting type not supported, please choose either 'Individual' or 'Polynomial'.")
            
        # Add free-parameters for a constant RMS background noise and constant Sersic poisson uncertainty scaling.
        ndim+=2

        return ndim
    

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
                keys.append(key+f'_C0')
                poly_order = param_prior_traits[2]
                for i in range(1,poly_order+1):
                    keys.append(key+f'_C{i}')

            else:
                raise Exception("Fitting type not supported, please choose either 'Individual' or 'Polynomial'.")

        # Add free-parameters for a constant RMS background noise and constant Sersic poisson uncertainty scaling.
        keys.append(f'rms')
        keys.append(f'rms_sersic')
        
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
                count+=1
                poly_order = param_prior_traits[2]

                # Calculate the conditional priors for higher order polynomial coefficients.
                for i in range(1,poly_order+1):

                    prev_order = i-1
                    prev_coeffs = theta[count-(prev_order+1):count]

                    polywavs = np.power(self.wavs[-1]-self.wavs[0], np.arange(prev_order+1))
                    prev_comps = prev_coeffs * polywavs
                    prev_comps_summed = np.sum(prev_comps)
                           
                    min = (param_prior_dist[0] - prev_comps_summed) / (np.power(self.wavs[-1]-self.wavs[0], i))
                    max = (param_prior_dist[1] - prev_comps_summed) / (np.power(self.wavs[-1]-self.wavs[0], i))
                    theta[count] = (theta[count] * (max-min)) + min
                    count+=1

            else:
                raise Exception("Fitting type not supported, please choose either 'Individual' or 'Polynomial'.")

        # Add free-parameters for a constant RMS background noise and constant Sersic poisson uncertainty scaling.
        theta[count] = theta[count]
        count+=1
        theta[count] = theta[count]*100

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
