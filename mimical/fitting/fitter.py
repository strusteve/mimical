import numpy as np
import matplotlib.pyplot as plt
import petrofit as pf
from astropy.convolution.utils import discretize_model
import corner
from nautilus import Sampler
import time
import os
from astropy.modeling import models
import pandas as pd
from tqdm import tqdm


from .priorHandler import priorHandler
from ..plotting import plotter

from ..utils import filter_set



dir_path = os.getcwd()
if not os.path.isdir(dir_path + "/mimical"):
    os.system('mkdir ' + dir_path + "/mimical")
    os.system('mkdir ' + dir_path + "/mimical/plots")
    os.system('mkdir ' + dir_path + "/mimical/posteriors")
    os.system('mkdir ' + dir_path + "/mimical/cats")



class mimical(object):
    """ Mimical is an intensity modelling code for multiply-imaged objects, 
    performing simultaenous Bayseian inference of model parameters via the 
    nested sampling algorithm. Mimical supports any astropy 2D model, and 
    supports user defined parameter polynomial depenency with image wavelength.

    Parameters
    ----------

    id : str
        An ID for the fitting run. Only really used for output files.

    images : array
        A 3D array of image data with slices for each filter. Each image
        must be the same shape.

    filt_list : str or list
        A list of path strings to the filter transmission curve files, relative
        to the current working directory. Must be in ascending order with effective wavelength.

    psfs : array
        A 3D array of normalised PSF images with slices for each filter. Each PSF image
        must be the same shape.

    astropy_model : array
        Astropy Fittable2DModel used to model the image data. The subsequent prior must include
        only and all parameters in the astropy_model.parameters variable, as well as a 'psf_pa' parameter.

    user_prior : dict
        The user specified prior which set out the priors for the model parameters
        and passes information about whether to let these vary for each filter or
        whether they follow an order-specified polynomial relationship.
    """


    def __init__(self, id, images, filt_list, psfs, user_prior, astropy_model=models.Sersic2D(), pool=None, timeout=3600):
        
        # Helper when only one image is passed
        if len(images.shape)==2:
            images = np.array(([images]))
        if (type(filt_list).__name__=='str') | (type(filt_list).__name__=='str_'):
            filt_list = [filt_list]
        if len(psfs.shape)==2:
            psfs = np.array(([psfs]))

        self.id = id
        print(f"Fitting object {self.id}")
        self.images = images
        self.psfs = psfs
        self.user_prior = user_prior
        self.astropy_model = astropy_model

        # Using the filter files, find the name of the filters and the effective wavelengths.
        self.filter_names = [x.split('/')[-1] for x in filt_list]
        self.wavs = filter_set([dir_path+'/'+x for x in filt_list]).eff_wavs / 1e4

        # Initiate the prior handler object, used to parse and translate priors and parameters.
        self.prior_handler = priorHandler(user_prior, self.filter_names, self.wavs)

        # Translate user specified prior into a prior parseable by nautlius.
        self.fitter_prior = self.prior_handler.translate()
        print(f"Fitting with parameters: {self.fitter_prior.keys}")

        self.t0 = time.time()
        self.calls = 0

        self.timeout=timeout
        self.pool = pool





    def lnlike(self, param_dict):
        """ Returns the log-likelihood for a given parameter vector. """

        # Translate parameter vector into model parameters in each filter.
        pars = self.prior_handler.revert(param_dict)

        # Define empty arrays for models and rms images.
        models = np.zeros_like(self.images)
        rms = np.zeros_like(self.images)
    
        # Loop over filters
        for i in range(len(self.wavs)):
            # Update the model and evaluate over a pixel grid.
            self.convolved_models[i].parameters = pars[i]
            model = discretize_model(model=self.convolved_models[i], 
                                     x_range=[0,self.images[i].shape[1]], 
                                     y_range=[0,self.images[i].shape[0]], 
                                     mode='center')

            # If, for whatever reason, the model has NaNs, set to zero and blow up errors.
            if np.isnan(np.sum(model)):
                models[i] = np.zeros_like(model)
                rms[i] = np.zeros_like(model) + 1e99

            # Else, append to respective arrays.
            else:
                models[i] = model
                rms[i] = param_dict['rms'] + (param_dict['rms_sersic']*np.sqrt(np.abs(model)))
                #rms[i] += 1.483 * median_abs_deviation(self.images[i].flatten())


        # Broadcast the 3D data and model arrays and sum through the resulting 3D log-likelihood array.
        log_like_array = np.log((1/(np.sqrt(2*np.pi*(rms**2))))) + ((-(self.images - models)**2) / (2*(rms**2)))
        log_like = np.sum(log_like_array.flatten())

        # Print calls
        self.calls+=1
        if not self.calls % 100:
            #print('Time % 100: ' + str(time.time()-self.t0))
            self.t0 = time.time()

        return(log_like)




    def fit(self):
        """ Runs the Nautlius sampler to fit models, and processes its output. """

        # Define empty models for each filter
        sersic_model = self.astropy_model
        self.convolved_models = []
        for i in range(len(self.wavs)):
            self.convolved_models.append(pf.PSFConvolvedModel2D(sersic_model, psf=self.psfs[i], oversample=(self.images.shape[2]/2, self.images.shape[1]/2, 15, 10)))

        # Check that the user specified prior contains the same parameters as the user specified model.
        if list(self.convolved_models[0].param_names) != list(self.user_prior.keys()):
            raise Exception("Prior labels do not match model parameters.")
        

        if os.path.isfile(dir_path+'/mimical/posteriors' + f'/{self.id}.txt'):
            self.success = True
            self.samples = pd.read_csv(dir_path+'/mimical/posteriors' + f'/{self.id}.txt', delimiter=' ').to_numpy()
            fit_dic = dict(zip((np.array((list(self.fitter_prior.keys)))+"_50").tolist(), np.median(self.samples, axis=0).tolist()))
            print(f"Loading existing posterior at " + dir_path + '/mimical/posteriors' + f'/{self.id}.txt')
            self.save_cat()
            return fit_dic


        # Run sampler
        t0 = time.time()
        #sampler = Sampler(self.fitter_prior, self.lnlike, n_live=400, filepath= dir_path+'/bogout'+f'/{self.id}.hdf5', resume=True, pool=self.pool)
        sampler = Sampler(self.fitter_prior, self.lnlike, n_live=400, pool=self.pool)
        self.success = sampler.run(verbose=True, timeout=self.timeout)
        print(f"Sampling time (minutes): {(time.time()-t0)/60}")

        if self.success != True:
            print("Sampling failed (timeout).")
            return {}
    
        else:
            # Sample the posterior information
            points, log_w, log_l = sampler.posterior()

            # Plot and save the corner plot
            corner.corner(points, weights=np.exp(log_w), bins=20, labels=np.array(self.fitter_prior.keys), color='purple', plot_datapoints=False, range=np.repeat(0.999, len(self.fitter_prior.keys)))
            plt.savefig(dir_path+'/mimical/plots' + f'/{self.id}_corner.pdf', bbox_inches='tight')

            # Sample an appropriately weighted posterior for representative samples.
            n_post = 10000
            indices = np.random.choice(np.arange(points.shape[0]), size = n_post, p=np.exp(log_w))
            self.samples = points[indices]
            samples_df = pd.DataFrame(data=self.samples, columns=self.fitter_prior.keys)
            samples_df.to_csv(dir_path+'/mimical/posteriors' + f'/{self.id}.txt', sep=' ', index=False)

            # Return the median-parameter model
            fit_dic = dict(zip((np.array((list(self.fitter_prior.keys)))+"_50").tolist(), np.median(self.samples, axis=0).tolist()))

            print("Sampling finished successfully.")

            self.save_cat()

            return fit_dic
        
    
    def save_cat(self):
        """ Saves the 16th/50th/84th percentiles of user prior parameter posteriors for each filter. """

        user_samples = np.zeros((self.samples.shape[0], len(self.wavs), len(self.user_prior.keys())))

        # Translates fitter samples into model parameter samples
        print("Computing model parameter posteriors...")
        for j in tqdm(range(self.samples.shape[0])):
            param_dict = dict(zip(list(self.fitter_prior.keys), self.samples[j]))
            pars = self.prior_handler.revert(param_dict)
            user_samples[j] = pars

        # Calculate percentiles
        quantiles = np.percentile(user_samples, q=(16, 50, 84), axis=0)

        # Save to .csv table
        dic = {}
        for j in range(len(self.filter_names)):
            for i in range(len(self.user_prior.keys())):
                key = list(self.user_prior.keys())[i]
                dic[key + "_" + self.filter_names[j] + "_16"] = [quantiles[0, j, i]]
                dic[key + "_" + self.filter_names[j] + "_50"] = [quantiles[1, j, i]]
                dic[key + "_" + self.filter_names[j] + "_84"] = [quantiles[2, j, i]]

        df = pd.DataFrame(dic)
        df.to_csv(dir_path+'/mimical/cats' + f'/{self.id}.csv', index=False)


    

    def plot_model(self, type='median'):
        """ Wrapper to plot models. """

        if self.success != True:
            print(f'Sampling failed, cannot plot model for {self.id}.')

        else:
            if type=='median':
                # Plot and save the median fit
                plotter().plot_median(self.images, self.wavs, self.convolved_models, self.samples, list(self.fitter_prior.keys), self.prior_handler, self.filter_names)
                plt.savefig(dir_path+'/mimical/plots' + f'/{self.id}_median_model.pdf', bbox_inches='tight', dpi=500)
            elif type=='median-param':
                # Plot and save the median-parameter fit
                plotter().plot_median_param(self.images, self.wavs, self.convolved_models, self.samples, list(self.fitter_prior.keys), self.prior_handler, self.filter_names)
                plt.savefig(dir_path+'/mimical/plots' + f'/{self.id}_median_param_model.pdf', bbox_inches='tight', dpi=500)

   
