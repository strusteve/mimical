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
from dynesty import DynamicNestedSampler
from dynesty.pool import Pool
from astropy.io import fits
from astropy.io import ascii

from .prior_handler import priorHandler
from ..plotting import plotter
from ..utils import filter_set
from ..utils import create_contmaps

dir_path = os.getcwd()
if not os.path.isdir(dir_path + "/mimical"):
    os.system('mkdir ' + dir_path + "/mimical")
    os.system('mkdir ' + dir_path + "/mimical/plots")
    os.system('mkdir ' + dir_path + "/mimical/posteriors")
    os.system('mkdir ' + dir_path + "/mimical/cats")

install_dir = os.path.dirname(os.path.realpath(__file__))
sextractor_dir = (install_dir + "/utils/sextractor_config").replace("/fitting","")



class fit(object):
    """ Fit a singly- or multiply-imaged object with a 2D model via Bayesian inference.

    Parameters
    ----------

    id : str
        An ID for the fitting run. Only really used for output files.

    images : ndarray or list of ndarray
        An image array or list of image arrays with elements for each filter.

    filt_list : str or list of str
        A path string  or list of path strings to the filter transmission curve files, relative
        to the current working directory.

    psfs : ndarray or list of ndarray
        A PSF image array or list of PSF image arrays with elements for each filter.

    mimical_prior : dict
        The user specified prior which set out the priors for the model parameters
        and passes information about whether to let these vary for each filter or
        whether they follow an order-specified polynomial relationship.

    astropy_model : array
        Astropy Fittable2DModel used to model the image data. The subsequent prior must include
        only and all parameters in the astropy_model.parameters variable, as well as a 'psf_pa' parameter.\
        
    pool : none or int
        Number of cores to parallelise likelihood calculations to.

    sampler : str
        Which sampler to use. Choice of Nautilus or Dynesty

    oversample_boxlength : int
        Width of box about image center to oversample within.

    oversample_factor : int
        Factor by which to oversample the central box.

    sextractor_clean : bool
        Whether or not to let sextractor clean the input images of contaminants.

    sextractor_target_maxdistancepix : str or float
        The distance after which the closest detected source is considered a contaminant.
        Necessary for images in which the target is undetected.
    """



    def __init__(self, id, images, filt_list, psfs, mimical_prior, 
                 astropy_model=models.Sersic2D(), pool=None, sampler='Nautilus', 
                 oversample_boxlength=15, oversample_factor=10, sextractor_clean=False,
                 sextractor_target_maxdistancepix='default', dilute=False, dilute_radius=6, runtag=''):
        
        # Start the clock
        self.genesis = time.time()

        # Set fitting ID
        self.id = id
        print(f"Fitting object {self.id}.")

        # Helper if only one image is being fitted
        if type(images).__name__ == 'ndarray':
            filt_list = [filt_list]
            self.images = [images]
            self.psfs = [psfs]
            self.contmaps = [np.ones_like(images)]
        elif type(images).__name__ == 'list':
            self.images = images
            self.filt_list = filt_list
            self.psfs = psfs
            self.contmaps = [np.ones_like(x) for x in self.images]
        else:
            raise Exception("Images must be either an ndarray or list of ndarrays")

        # Set the Mimical prior
        self.mimical_prior = mimical_prior

        # Set keyword arguments
        self.astropy_model = astropy_model
        self.pool = pool
        self.sampler = sampler
        self.oversample_boxlength = oversample_boxlength
        self.oversample_factor = oversample_factor
        self.sextractor_clean = sextractor_clean
        self.runtag = runtag


        # Find the names and effective wavelengths of image filters
        self.filter_names = [x.split('/')[-1] for x in filt_list]
        self.wavs = filter_set([dir_path+'/'+x for x in filt_list]).eff_wavs / 1e4

        # Sort the image information in order of ascending wavelength
        if not len(self.wavs)==1:
            sorter = np.argsort(self.wavs)
            self.wavs = [self.wavs[x] for x in sorter]
            self.filter_names = [self.filter_names[x] for x in sorter]
            self.images = [self.images[x] for x in sorter]
            self.filt_list = [self.filt_list[x] for x in sorter]
            self.psfs = [self.psfs[x] for x in sorter]
            self.contmaps = [self.contmaps[x] for x in sorter]
            for key in self.mimical_prior.keys():
                if (type(self.mimical_prior[key][0]).__name__ == 'list'):
                    self.mimical_prior[key] = ([self.mimical_prior[key][0][x] for x in sorter], *self.mimical_prior[key][1:])
                elif (type(self.mimical_prior[key][0]).__name__ == 'ndarray'):
                    raise Exception("If specifying values for each filter, please pass these in with a list, not an ndarray.")
                else:
                    continue

        # Initiate the prior handler object, used to parse and translate priors and parameters
        self.prior_handler = priorHandler(mimical_prior, self.filter_names, self.wavs, self.images, self.runtag, self.id)
        self.sampler_prior_keys = self.prior_handler.generate_sampler_prior_keys()
        print(f"Fitting {self.prior_handler.nmodel}-parameter models with" +
                      f"{self.prior_handler.nparam}-parameter Mimical fit with dimensionality" +
                      f"{self.prior_handler.ndim}.")

        # Define dummy models for each filter, quicker to update parameters than create new models in the likelihood
        sersic_model = self.astropy_model
        self.convolved_models = []
        for i in range(len(self.wavs)):
            self.convolved_models.append(pf.PSFConvolvedModel2D(sersic_model, psf=self.psfs[i], oversample=(self.images[i].shape[1]/2, self.images[i].shape[0]/2, self.oversample_boxlength, self.oversample_factor)))

        # Set the SExtractor criterion for definining the closest object as noise
        if sextractor_target_maxdistancepix=='default':
            self.target_maxdistancepix = self.images[0].shape[0]/5
        else:
            self.target_maxdistancepix = sextractor_target_maxdistancepix

        self.dilute = dilute
        self.dilute_radius = dilute_radius

        # Code-timing interface
        self.calls = 0
        self.calltime = 0
        self.bugtime = 0

    
    def get_residuals(self, param_dict):

        # Translate unit cube prior sample into Mimical prior sample
        reverted = self.prior_handler.revert(param_dict)

        # Check if sampled model paramters are all within bounds, if not - blow up
        for j in range(len(list(self.mimical_prior.keys()))):
            bounds = self.mimical_prior[list(self.mimical_prior.keys())[j]][0]
            if type(bounds).__name__ == "tuple":
                if (any(reverted[:,j] < bounds[0])) | (any(reverted[:,j] > bounds[1])): return -9.99*10**99
            else: continue
        
        # Pull out model parameters, as well as supplementary RMS and counts-per-flux parameters
        modelpars = reverted[:,:self.prior_handler.nmodel]
        rmsarr = reverted[:,self.prior_handler.nmodel]
        cpfarr = reverted[:,self.prior_handler.nmodel+1]

        # If user provides RMS values, override prior sample - necessary to recover full arrays
        if not (type(self.mimical_prior['rms'][0]).__name__ == 'tuple'):
            if (type(self.mimical_prior['rms'][0]).__name__ == 'list'):
                if (type(self.mimical_prior['rms'][0][0]).__name__ == 'ndarray'):
                    rmsarr = self.mimical_prior['rms'][0]
            elif (len(self.wavs) == 1) & ((type(self.mimical_prior['rms'][0]).__name__ == 'ndarray')):
                rmsarr = [self.mimical_prior['rms'][0]]
            # If user wants mimical to infer RMS from image background, do so
            elif (type(self.mimical_prior['rms'][0]).__name__ == 'str') & (self.mimical_prior['rms'][0] == "Infer"):
                    if not self.sextractor_clean: raise Exception("If using Infer special type for RMS, must set sextractor_clean=True.")

        # If user provides counts-per-flux parameters, override prior sample - necessary to recover full arrays
        if not (type(self.mimical_prior['counts_per_flux'][0]).__name__ == 'tuple'):
            if (type(self.mimical_prior['counts_per_flux'][0]).__name__ == 'list'):
                if ((type(self.mimical_prior['counts_per_flux'][0][0]).__name__ == 'ndarray')):
                    cpfarr = self.mimical_prior['counts_per_flux'][0]
            elif (len(self.wavs) == 1) & (type(self.mimical_prior['counts_per_flux'][0]).__name__ == 'ndarray'):
                cpfarr = [self.mimical_prior['counts_per_flux'][0]]

        residuals_arr = []
        sigma_arr = []

        # Loop over filters
        for i in range(len(self.wavs)):

            # Update the model and evaluate over a pixel grid.
            self.convolved_models[i].parameters = modelpars[i]
            model = discretize_model(model=self.convolved_models[i], 
                                        x_range=[0,self.images[i].shape[1]], 
                                        y_range=[0,self.images[i].shape[0]], 
                                        mode='center')

            # If, for whatever reason, the model has NaNs, set to zero and blow up errors.
            if np.isnan(np.sum(model)):
                model = np.zeros_like(model)
                sigma = np.zeros_like(model) + 1e99
                print('Unphysical model detected.')

            # Else, append to respective arrays.
            else:
                sigma = np.sqrt(rmsarr[i]**2 + ((cpfarr[i]**(-1/2))*np.sqrt(np.abs(model)))**2)
                #sigma = np.zeros_like(model) + rmsarr[i]
                if type(sigma).__name__ == 'ndarray':
                    sigma[(~np.isfinite(sigma)) | (sigma==0)] = 1e99

            # Calculate the 3D mask
            contmask_3D = self.contmaps[i] == 1

            # Calculate the filter specific likelihood and add to total
            residuals = self.images[i][contmask_3D].flatten() - model[contmask_3D].flatten()

            residuals_arr.extend(residuals)
            sigma_arr.extend(sigma[contmask_3D].flatten())

        return np.array((residuals_arr)), np.array((sigma_arr))

        
            

    def lnlike(self, param_dict):
        """ Returns the log-likelihood for a given parameter vector. """

        # Set likelihood clock
        time0 = time.time()

        residuals, sigma = self.get_residuals(param_dict)        

        norm = np.log((1/(np.sqrt(2*np.pi*(sigma**2)))))
        log_like_array = norm + ((-(residuals)**2) / (2*(sigma**2)))
        log_like = np.sum(log_like_array)

        '''
        self.calls += 1
        self.calltime += time.time()-time0
        if self.calls % 1000 == 0:
            print(f"Average call time: {1000*(self.calltime/self.calls)}")
            print(f"Average bug time: {(1000*(self.bugtime/self.calls))} ")
            print(f"Average bug time: {((self.bugtime/self.calls)/(self.calltime/self.calls))*100} %")
            print(' ')
        '''

        return(log_like)



    def run(self):
        """ Runs the nested sampler to sample models, and processes its output. """
    
        # Create sub-directories for specific catalogue runs, if they don't already exist
        if not os.path.isdir(dir_path + f"/mimical/posteriors{self.runtag}"):
            os.system('mkdir ' + dir_path + f"/mimical/plots{self.runtag}")
            os.system('mkdir ' + dir_path + f"/mimical/posteriors{self.runtag}")

        # Run SExtractor cleaning step if desired
        if self.sextractor_clean == True:
            self.contmaps = create_contmaps(self.id, self.wavs, self.images, self.filter_names, self.contmaps, self.target_maxdistancepix, self.dilute, self.dilute_radius, self.runtag)

        # Check that the user specified prior contains all the parameters as the user specified 2D model
        if list(self.convolved_models[0].param_names) != list(self.mimical_prior.keys())[:-2]:
            raise Exception("Prior labels do not match model parameters.")
        
        # Check if a posterior already exists for the object being fitted, if so load it
        if os.path.isfile(dir_path+f'/mimical/posteriors{self.runtag}' + f'/{self.id}_samples.txt'):
            self.points = np.loadtxt(dir_path+f"/mimical/posteriors{self.runtag}/{self.id}_points.txt")
            self.log_w = np.loadtxt(dir_path+f"/mimical/posteriors{self.runtag}/{self.id}_logw.txt")
            self.log_l = np.loadtxt(dir_path+f"/mimical/posteriors{self.runtag}/{self.id}_logl.txt")
            self.samples = pd.read_csv(dir_path+f'/mimical/posteriors{self.runtag}' + f'/{self.id}_samples.txt', delimiter=' ').to_numpy()
            print(f"Loading existing posterior at " + dir_path + f'/mimical/posteriors{self.runtag}' + f'/{self.id}.txt')
            self.save_output()
        
        # If not, sample
        else:
            # Set the sampler prior
            self.sampler_prior = self.prior_handler.sampler_prior

            # Run sampling with Nautilus
            if self.sampler == 'Nautilus':
                t0 = time.time()
                sampler = Sampler(self.sampler_prior, self.lnlike, n_live=400, pool=self.pool, n_dim = self.prior_handler.ndim)
                sampler.run(verbose=True)
                print(f"Sampling time (minutes): {(time.time()-t0)/60}")
                self.points, self.log_w, self.log_l = sampler.posterior()

            # Run sampling with Dynesty
            elif self.sampler == 'Dynesty':
                t0 = time.time()
                if self.pool==None:
                    sampler = DynamicNestedSampler(self.lnlike, self.sampler_prior, ndim = self.prior_handler.ndim, nlive=400)
                    sampler.run_nested()
                else:
                    with Pool(self.pool, self.lnlike, self.sampler_prior) as pool:
                        sampler = DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim = self.prior_handler.ndim, nlive=400, pool=pool)
                        sampler.run_nested()
                print(f"Sampling time (minutes): {(time.time()-t0)/60}")
                results = sampler.results
                self.points, self.log_w = results.samples, np.log(results.importance_weights())

            else:
                raise Exception(f"Sampler {self.sampler} not supported. (Please choose either 'Nautilus' or 'Dynesty')")

            print("Sampling finished successfully.")

            self.save_output()


    def calc_chisq(self, param_dict):
        """ Calculates the Chisq value for the maximum likelihood model. """
        
        residuals, sigma = self.get_residuals(param_dict)
        chisq_arr = residuals**2 / sigma**2
        chisq = np.sum(chisq_arr)

        return chisq, len(chisq_arr)
    


    def save_output(self):
        """ Saves the 16th/50th/84th percentiles of user prior parameter posteriors for each filter. """

        # Save the sampled points and corresponding log-weights
        np.savetxt(dir_path+f"/mimical/posteriors{self.runtag}/{self.id}_points.txt", self.points)
        np.savetxt(dir_path+f"/mimical/posteriors{self.runtag}/{self.id}_logw.txt", self.log_w)
        np.savetxt(dir_path+f"/mimical/posteriors{self.runtag}/{self.id}_logl.txt", self.log_l)

        # Sample an appropriately weighted posterior for representative samples, and save
        n_post = 10000
        indices = np.random.choice(np.arange(self.points.shape[0]), size = n_post, p=np.exp(self.log_w))
        self.samples = self.points[indices]
        samples_df = pd.DataFrame(data=self.samples, columns=self.sampler_prior_keys)
        samples_df.to_csv(dir_path+f"/mimical/posteriors{self.runtag}/{self.id}_samples.txt", sep=' ', index=False)

        # Plot and save the corner plot
        corner.corner(self.points.T[self.prior_handler.samplemask].T, weights=np.exp(self.log_w), bins=20, labels=np.array(self.sampler_prior_keys)[self.prior_handler.samplemask], color='black', plot_datapoints=False, range=np.repeat(0.999, np.sum(self.prior_handler.samplemask)))
        plt.savefig(dir_path+f'/mimical/plots{self.runtag}/{self.id}_corner.pdf', bbox_inches='tight')

        # Define empty samples array and translate Mimical samples into model parameter samples
        samples_mimical = np.apply_along_axis(lambda param_vec: self.prior_handler.revert(param_vec).flatten(), 1, self.samples).reshape(self.samples.shape[0], len(self.wavs), len(self.mimical_prior.keys()))

        # Calculate percentiles
        quantiles = np.percentile(samples_mimical, q=(16, 50, 84), axis=0)
        
        # Create dataframe table
        dic = {"id":self.id}
        for j in range(len(self.wavs)):
            for i in range(len(self.mimical_prior.keys())):
                key = list(self.mimical_prior.keys())[i]
                dic[key + "_" + self.filter_names[j] + "_16"] = [quantiles[0, j, i]]
                dic[key + "_" + self.filter_names[j] + "_50"] = [quantiles[1, j, i]]
                dic[key + "_" + self.filter_names[j] + "_84"] = [quantiles[2, j, i]]
        chisq, numd = self.calc_chisq(self.points[np.argmax(self.log_l)])
        dic['datanum'] = numd
        dic['chisq'] = chisq
        df = pd.DataFrame(dic)

        # If not part of a catalogue fit, save individual
        if self.runtag=='':
            df.to_csv(dir_path+f'/mimical/cats/{self.id}.csv', index=False)

        # If part of a catalogue fit, either start a catalogue file or append to it.
        else:
            if not os.path.isfile(dir_path + f"/mimical/cats{self.runtag}.csv"):
                df.to_csv(dir_path+f'/mimical/cats{self.runtag}.csv', index=False)
            else:
                ridden = pd.read_csv(dir_path+f'/mimical/cats{self.runtag}.csv')
                ridden.index = ridden['id'].values.astype('str')
                if self.id not in ridden.index.values:
                    ridden.loc[self.id] = df.values[0]
                    ridden.to_csv(dir_path+f'/mimical/cats{self.runtag}.csv', index=False)
                else:
                    print('Object already written to catalogue.')



    def plot_model(self):
        """ Wrapper to plot models. """

        # Plot and save the maxL fit
        plotter().plot_best(self.images, self.wavs, self.convolved_models, self.points[np.argmax(self.log_l)], self.prior_handler, self.filter_names, self.contmaps)
        plt.savefig(dir_path+f'/mimical/plots{self.runtag}/{self.id}_best_model.pdf', bbox_inches='tight', dpi=500, transparent=True)

        # Plot the trends with wavelength if multiband fit
        if len(self.wavs) > 1:
            plotter().plot_trends(self.wavs, self.samples, self.prior_handler, self.mimical_prior)
            plt.savefig(dir_path+f'/mimical/plots{self.runtag}/{self.id}_trends.pdf', bbox_inches='tight', dpi=500, transparent=True)

        # Plot the errors used in fitting
        plotter().plot_errors(self.images, self.wavs, self.mimical_prior, self.convolved_models, self.points[np.argmax(self.log_l)], self.prior_handler, self.filter_names, self.contmaps)
        plt.savefig(dir_path+f'/mimical/plots{self.runtag}/{self.id}_errors.pdf', bbox_inches='tight', dpi=500, transparent=True)
    