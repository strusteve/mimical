from nautilus import Prior
from scipy.stats import norm
import numpy as np

class priorHandler(object):
    """ Contains functions for translating Mimical user priors into 
    Nautlius priors and translating Nautilus samples into model
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


    def translate(self):
        """ Translate a Mimical prior into a Nautilus prior."""
        # Initiate Nautlius prior
        prior = Prior()

        # Loop over model parameters
        for key in self.user_prior.keys():
            param_prior_traits = self.user_prior[key]
            param_prior_dist = param_prior_traits[0]
            param_fit_type = param_prior_traits[1]

            # If user specifies 'Individual', add a fitter free-parameter for each filter.
            if param_fit_type == "Individual":
                for i in range(len(self.wavs)):
                    prior.add_parameter(f'{key}_{self.filter_names[i]}', dist=param_prior_dist)
            
            # If user specifies 'Polynomial', add a fitter free-parameter for each coefficient. 
            # e.g., For order 0, only one free-parameter is included for the whole fitting run a.k.a constant between filters.
            # e.g., For order 1, two free-parameter are included for the whole fitting run a.k.a straight-line relationship  with effective wavelength.
            # The lowest wavelength is chosen as the origin.
            elif param_fit_type == "Polynomial":
                prior.add_parameter(key+f'_C0', dist=param_prior_dist)
                poly_order = param_prior_traits[2]
                higher_order_dist = norm(loc=0, scale=1) # Priors for gradients, higher-order coefficients.
                for i in range(1,poly_order+1):
                    prior.add_parameter(key+f'_C{i}', dist=higher_order_dist)

            else:
                raise Exception("Fitting type not supported, please choose either 'Individual' or 'Polynomial'.")

        # Add free-parameters for a constant RMS background noise and constant Sersic poisson uncertainty scaling.
        prior.add_parameter(f'rms', dist=(0,1))
        prior.add_parameter(f'rms_sersic', dist=(0,100))
        
        return prior


    def revert(self, param_dict):
        """ Translate a Nautilus sample into a sample of model parameters for each filter."""

        # Empty parameter array
        params_final = np.zeros((len(self.wavs), len(self.user_prior.keys())))

        # Loop over model parameters
        keys = list(self.user_prior.keys())
        for i in range(len(keys)):
            param_prior_traits = self.user_prior[keys[i]]
            param_fit_type = param_prior_traits[1]

            # If individual, add the Nautilus sample for each filter
            if param_fit_type == "Individual":
                for j in range(len(self.wavs)):
                    params_final[j,i] = param_dict[f"{keys[i]}_{self.filter_names[j]}"]
            
            # If polynomial, calculate the expected parameter in each filter given its effective wavlength
            elif param_fit_type == "Polynomial":
                poly_order = param_prior_traits[2]
                coeffs = np.zeros(poly_order+1)
                for k in range(poly_order+1):
                    coeffs[k] = param_dict[f"{keys[i]}_C{k}"]
                polywavs = np.power(np.tile(self.wavs-self.wavs[0], (poly_order+1,1)).T, np.arange(poly_order+1))
                comps = coeffs * polywavs
                comps_summed = np.sum(comps, axis=1)
                params_final[:,i] =comps_summed        

        return params_final
