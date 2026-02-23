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


dir_path = os.getcwd()
if not os.path.isdir(dir_path + "/mimical"):
    os.system('mkdir ' + dir_path + "/mimical")
    os.system('mkdir ' + dir_path + "/mimical/plots")
    os.system('mkdir ' + dir_path + "/mimical/posteriors")
    os.system('mkdir ' + dir_path + "/mimical/cats")

install_dir = os.path.dirname(os.path.realpath(__file__))
sextractor_dir = (install_dir + "/utils/sextractor_config").replace("/fitting","")


class fit(object):
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
                 sextractor_target_maxdistancepix='default'):

        print(' ')

        # Start the clock
        self.genesis = time.time()

        # Set passed variables
        self.id = id
        print(f"Fitting object {self.id}.")
        self.images = images
        self.filt_list = filt_list
        self.psfs = psfs
        self.user_prior = mimical_prior

        # Set defaulted values
        self.astropy_model = astropy_model
        self.pool = pool
        self.sampler = sampler
        self.oversample_boxlength = oversample_boxlength
        self.oversample_factor = oversample_factor
        self.sextractor_clean = sextractor_clean

        # Find the names and effective wavelengths of image filters
        self.filter_names = [x.split('/')[-1] for x in filt_list]
        self.wavs = filter_set([dir_path+'/'+x for x in filt_list]).eff_wavs / 1e4

        # Initiate the prior handler object, used to parse and translate priors and parameters.
        self.prior_handler = priorHandler(mimical_prior, self.filter_names, self.wavs)
        print(f"Fitting {self.prior_handler.nmodel}-parameter models with {self.prior_handler.nparam}-parameter Mimical fit with dimensionality {self.prior_handler.ndim}.")
        self.sampler_prior_keys = self.prior_handler.generate_sampler_prior_keys()
        print(self.sampler_prior_keys)
        # Set Sextractor criterion for definining closest object as noise
        if sextractor_target_maxdistancepix=='default':
            self.target_maxdistancepix = self.images.shape[1]/5
        else:
            self.target_maxdistancepix = sextractor_target_maxdistancepix
        
        # Initialise default segmentation maps
        self.segmaps = np.ones_like(self.images)


        # Define empty models for each filter
        sersic_model = self.astropy_model
        self.convolved_models = []
        for i in range(len(self.wavs)):
            self.convolved_models.append(pf.PSFConvolvedModel2D(sersic_model, psf=self.psfs[i], oversample=(self.images.shape[2]/2, self.images.shape[1]/2, self.oversample_boxlength, self.oversample_factor)))



    def create_segmaps_sextractor(self, runtag=''):
        """ Method for cleaning contaminated images with sextractor, overwrites images and segmentation maps. """
        print(runtag)
        # Make output directories for sextractor output
        if not os.path.isdir(dir_path + "/mimical/sextractor"):
            os.system('mkdir ' + dir_path + "/mimical/sextractor")
            os.system('mkdir ' + dir_path + "/mimical/sextractor/input_images")
            os.system('mkdir ' + dir_path + "/mimical/sextractor/cats")
            os.system('mkdir ' + dir_path + "/mimical/sextractor/segmaps")

        if not os.path.isdir(dir_path + f"/mimical/sextractor/cats{runtag}"):
            os.system('mkdir ' + dir_path + f"/mimical/sextractor/input_images{runtag}")
            os.system('mkdir ' + dir_path + f"/mimical/sextractor/cats{runtag}")
            os.system('mkdir ' + dir_path + f"/mimical/sextractor/segmaps{runtag}")


        # Save images passed to Mimical for passing to Sextractor
        for i in range(len(self.filter_names)):
            hdul = fits.HDUList()
            hdul.append(fits.ImageHDU(data=self.images[i]))
            hdul.writeto(f"{dir_path}/mimical/sextractor/input_images{runtag}/{self.id}_{self.filter_names[i]}.fits", overwrite=True)

        # Run Sextractor
        for i in range(len(self.filter_names)):
            os.system(f"sex {dir_path}/mimical/sextractor/input_images{runtag}/{self.id}_{self.filter_names[i]}.fits" +
                      f" -c {sextractor_dir}/jwst_default_segmap.config" +
                      f" -FILTER_NAME {sextractor_dir}/gauss_2.5_5x5.conv" +
                      f" -PARAMETERS_NAME {sextractor_dir}/default.param" +
                      f" -CATALOG_NAME {dir_path}/mimical/sextractor/cats{runtag}/{self.id}_{self.filter_names[i]}.cat" +
                      f" -CHECKIMAGE_NAME {dir_path}/mimical/sextractor/segmaps{runtag}/{self.id}_{self.filter_names[i]}.fits")
            
        # Loop over filters, load Sextractor catalogues and segmentation maps, determine any areas of contamination and set them to zero.

        for i in range(len(self.filter_names)):
            image = self.images[i]
            centre_x, centre_y = image.shape[1]/2, image.shape[0]/2
            cat = ascii.read(f"{dir_path}/mimical/sextractor/cats{runtag}/{self.id}_{self.filter_names[i]}.cat").to_pandas()
            cat['sep'] = np.sqrt( (cat['X_IMAGE']-centre_x)**2 +  (cat['Y_IMAGE']-centre_y)**2  )
            cat.index = cat['NUMBER'].values

            # If no objects found, leave segmap as ones.
            if len(cat)==0:
                continue

            else:
                segmap = fits.open(f"{dir_path}/mimical/sextractor/segmaps{runtag}/{self.id}_{self.filter_names[i]}.fits")[0].data

                # If only one object found
                if len(cat)==1:
                    obj_of_interest = cat.iloc[0]

                # If multiple objects found
                else:
                    obj_of_interest = cat.loc[cat['NUMBER'].values[np.argmin(cat['sep'])]]

                # If closest object is not near centre, cut it / others
                if obj_of_interest['sep'] > self.target_maxdistancepix:
                    segmap += 1
                    segmap[segmap!=1] = 0
                    self.segmaps[i] = segmap

                # If closest object is near centre, cut all else
                else:
                    segmap += 1
                    segmap[(segmap!=1) & (segmap!=obj_of_interest['NUMBER']+1)] = 0
                    segmap[segmap!=0] = 1
                    self.segmaps[i] = segmap


        

    def lnlike(self, param_dict):
        """ Returns the log-likelihood for a given parameter vector. """

        # Translate parameter vector into model parameters in each filter.
        reverted = self.prior_handler.revert(param_dict)

        # Check within bounds
        for j in range(len(list(self.user_prior.keys()))):
            bounds = self.user_prior[list(self.user_prior.keys())[j]][0]
            if (any(reverted[:,j] < bounds[0])) | (any(reverted[:,j] > bounds[1])):
                return -9.99*10**99
            

        modelpars = reverted[:,:self.prior_handler.nmodel]
        rmsarr = reverted[:,self.prior_handler.nmodel]
        ftcarr = reverted[:,self.prior_handler.nmodel+1]


        # Define empty arrays for models and rms images.
        models = np.zeros_like(self.images)
        rms = np.zeros_like(self.images)
    
        # Loop over filters
        for i in range(len(self.wavs)):
            # Update the model and evaluate over a pixel grid.
            self.convolved_models[i].parameters = modelpars[i]
            model = discretize_model(model=self.convolved_models[i], 
                                     x_range=[0,self.images[i].shape[1]], 
                                     y_range=[0,self.images[i].shape[0]], 
                                     mode='center')
            
            model = model * self.segmaps[i]

            # If, for whatever reason, the model has NaNs, set to zero and blow up errors.
            if np.isnan(np.sum(model)):
                models[i] = np.zeros_like(model)
                rms[i] = np.zeros_like(model) + 10**99

            # Else, append to respective arrays.
            else:
                models[i] = model
                rms[i] = np.sqrt(rmsarr[i]**2 + ((ftcarr[i]**(-1/2))*np.sqrt(np.abs(model)))**2)

        # Broadcast the 3D data and model arrays and sum through the resulting 3D log-likelihood array.
        segmask_3D = self.segmaps == 1
        log_like_array = np.log((1/(np.sqrt(2*np.pi*(rms[segmask_3D].flatten()**2))))) + ((-(self.images[segmask_3D].flatten() - models[segmask_3D].flatten())**2) / (2*(rms[segmask_3D].flatten()**2)))
        log_like = np.sum(log_like_array)

        return(log_like)


    def run(self, runtag=''):
        """ Runs the nested sampler to sample models, and processes its output. """


        # Check that the user specified prior contains the same parameters as the user specified model.
        if list(self.convolved_models[0].param_names) != list(self.user_prior.keys())[:-2]:
            raise Exception("Prior labels do not match model parameters.")
        
        if os.path.isfile(dir_path+f'/mimical/posteriors{runtag}' + f'/{self.id}_samples.txt'):
            self.points = np.loadtxt(dir_path+f"/mimical/posteriors{runtag}/{self.id}_points.txt")
            self.log_w = np.loadtxt(dir_path+f"/mimical/posteriors{runtag}/{self.id}_logw.txt")
            self.samples = pd.read_csv(dir_path+f'/mimical/posteriors{runtag}' + f'/{self.id}_samples.txt', delimiter=' ').to_numpy()
            print(f"Loading existing posterior at " + dir_path + f'/mimical/posteriors{runtag}' + f'/{self.id}.txt')
            self.save_output(runtag=runtag)
        
        else:
            # Run sextractor cleaning step if desired
            if self.sextractor_clean == True:
                self.create_segmaps_sextractor(runtag=runtag)

            self.sampler_prior = self.prior_handler.translate()

            # Run sampling with Nautilus
            if self.sampler == 'Nautilus':
                t0 = time.time()
                sampler = Sampler(self.sampler_prior, self.lnlike, n_live=400, pool=self.pool, n_dim = self.prior_handler.nparam)
                sampler.run(verbose=True)
                print(f"Sampling time (minutes): {(time.time()-t0)/60}")
                self.points, self.log_w, log_l = sampler.posterior()

            # Run sampling with Dynesty
            elif self.sampler == 'Dynesty':
                t0 = time.time()
                if self.pool==None:
                    sampler = DynamicNestedSampler(self.lnlike, self.sampler_prior, ndim = self.prior_handler.nparam, nlive=400)
                    sampler.run_nested()
                else:
                    with Pool(self.pool, self.lnlike, self.sampler_prior) as pool:
                        sampler = DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim = self.prior_handler.nparam, nlive=400, pool=pool)
                        sampler.run_nested()
                print(f"Sampling time (minutes): {(time.time()-t0)/60}")
                results = sampler.results
                self.points, self.log_w = results.samples, np.log(results.importance_weights())

            else:
                raise Exception(f"Sampler {self.sampler} not supported. (Please choose either 'Nautilus' or 'Dynesty')")

            print("Sampling finished successfully.")

            self.save_output(runtag=runtag)

        
    
    def save_output(self, runtag=''):
        """ Saves the 16th/50th/84th percentiles of user prior parameter posteriors for each filter. """

        if not os.path.isfile(dir_path + f"/mimical/cats{runtag}.csv"):
            os.system('mkdir ' + dir_path + f"/mimical/plots{runtag}")
            os.system('mkdir ' + dir_path + f"/mimical/posteriors{runtag}")

        np.savetxt(dir_path+f"/mimical/posteriors{runtag}/{self.id}_points.txt", self.points)
        np.savetxt(dir_path+f"/mimical/posteriors{runtag}/{self.id}_logw.txt", self.log_w)

        # Save samples to file
        # Sample an appropriately weighted posterior for representative samples.
        n_post = 10000
        indices = np.random.choice(np.arange(self.points.shape[0]), size = n_post, p=np.exp(self.log_w))
        self.samples = self.points[indices]
        samples_df = pd.DataFrame(data=self.samples, columns=self.sampler_prior_keys)
        samples_df.to_csv(dir_path+f"/mimical/posteriors{runtag}/{self.id}_samples.txt", sep=' ', index=False)

        # Plot and save the corner plot
        corner.corner(self.points, weights=np.exp(self.log_w), bins=20, labels=np.array(self.sampler_prior_keys), color='black', plot_datapoints=False, range=np.repeat(0.999, len(self.sampler_prior_keys)))
        plt.savefig(dir_path+f'/mimical/plots{runtag}/{self.id}_corner.pdf', bbox_inches='tight')

        # Return the median-parameter model
        user_samples = np.zeros((self.samples.shape[0], len(self.wavs), len(self.user_prior.keys())))

        # Translates fitter samples into model parameter samples
        print("Computing model parameter posteriors...")
        for j in tqdm(range(self.samples.shape[0])):
            param_dict = self.samples[j]
            pars = self.prior_handler.revert(param_dict)
            user_samples[j] = pars

        # Calculate percentiles
        quantiles = np.percentile(user_samples, q=(16, 50, 84), axis=0)


        # Save to .csv table
        dic = {"id":self.id}
        for j in range(len(self.filter_names)):
            for i in range(len(self.user_prior.keys())):
                key = list(self.user_prior.keys())[i]
                dic[key + "_" + self.filter_names[j] + "_16"] = [quantiles[0, j, i]]
                dic[key + "_" + self.filter_names[j] + "_50"] = [quantiles[1, j, i]]
                dic[key + "_" + self.filter_names[j] + "_84"] = [quantiles[2, j, i]]

        df = pd.DataFrame(dic)

        if runtag=='':
            df.to_csv(dir_path+f'/mimical/cats/{self.id}.csv', index=False)

        else:
            if not os.path.isfile(dir_path + f"/mimical/cats{runtag}.csv"):
                df.to_csv(dir_path+f'/mimical/cats{runtag}.csv', index=False)
            else:
                ridden = pd.read_csv(dir_path+f'/mimical/cats{runtag}.csv')
                ridden.index = ridden['id'].values
                ridden.loc[self.id] = df.values[0]
                ridden.to_csv(dir_path+f'/mimical/cats{runtag}.csv', index=False)


    def plot_model(self, type='median', runtag=''):
        """ Wrapper to plot models. """

        if type=='median':
            # Plot and save the median fit
            plotter().plot_median(self.images, self.wavs, self.convolved_models, self.samples, self.prior_handler, self.filter_names, self.segmaps)
            plt.savefig(dir_path+f'/mimical/plots{runtag}/{self.id}_median_model.pdf', bbox_inches='tight', dpi=500)
        elif type=='median-param':
            # Plot and save the median-parameter fit
            plotter().plot_median_param(self.images, self.wavs, self.convolved_models, self.samples, self.prior_handler, self.filter_names, self.segmaps)
            plt.savefig(dir_path+f'/mimical/plots{runtag}/{self.id}_median_param_model.pdf', bbox_inches='tight', dpi=500)

