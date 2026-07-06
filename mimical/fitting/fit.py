import numpy as np
import matplotlib.pyplot as plt
import corner
from nautilus import Sampler
import time
import os
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator as RGI

from ..priors.prior_handler import priorHandler
from ..plotting import plotting
from ..utils import filter_set
from ..utils import create_contmaps
from ..utils import make_oversampling_table
from ..models.submodels import Sersic
from ..models.imagemodel import ImageModel

dir_path = os.getcwd()
if not os.path.isdir(dir_path + "/mimical_output"):
    os.system('mkdir ' + dir_path + "/mimical_output")
    os.system('mkdir ' + dir_path + "/mimical_output/plots")
    os.system('mkdir ' + dir_path + "/mimical_output/posteriors")
    os.system('mkdir ' + dir_path + "/mimical_output/cats")

install_dir = os.path.dirname(os.path.realpath(__file__))
sextractor_dir = (install_dir + "/utils/sextractor_config").replace("/fitting",
                                                                    "")
tabledir = (install_dir + "/utils/oversampling_table").replace("/fitting", "")


class fit(object):
    """ Fit a singly- or multiply-imaged object with a 2D model.

    Parameters
    ----------

    id : str
        An ID for the fitting run. Only really used for output files.

    images : list
        List of 2D images with slices for each filter. Must be the same shape
        and pixel scale.

    filt_list : list
        A list of path strings to the filter transmission curve files,
        relative to the current working directory.

    psfs : list
        A list of PSF images with slices for each filter, normalised to one.

    mimical_prior : dict
        The user specified prior which set out the priors for the model
        parameters and passes information about whether to let these vary for
        each filter or whether they follow an order-specified polynomial
        relationship.

    submodel : mimical.submodel
        The mimical submodel used in sampling. Currently only 'Sersic' is
        supported.

    se_clean : bool
        Whether or not to let SourceExtractor clean the input images of
        contaminants. Must allow 'sex' command via terminal.

    se_maxdist: str or float
        The distance after which the closest detected source is considered a
        contaminant. Necessary for images in which the target is undetected.

    dilute : bool
        Whether or not to apply a circular miminum filter over the
        contamination map to dilute it.

    dilute_radius : int
        If dilute is 'True', apply minimim filter with radius 'dilute_radius'
        over the contamination map.
    """

    def __init__(self, id, images, filt_list, psfs, mimical_prior,
                 submodel=Sersic, se_clean=False, se_maxdist='default',
                 dilute=True, dilute_radius=5, runtag=''):

        self.genesis = time.time()

        # Set positional arguments
        self.id = id
        self.images = np.array((images))
        self.filt_list = filt_list
        if psfs is not None:
            self.psfs = np.array((psfs))
        else:
            self.psfs = psfs
        self.mimical_prior = mimical_prior
        print(f"Fitting object {self.id}.")

        # Set keyword arguments
        self.submodel = submodel
        self.se_clean = se_clean
        self.runtag = runtag

        # Find the names and effective wavelengths of image filters
        self.filter_names = [x.split('/')[-1] for x in self.filt_list]
        filt_set = filter_set([dir_path+'/'+x for x in self.filt_list])
        self.wavs = (filt_set.eff_wavs / 1e4)

        """
        # If single image fit, make mimical prior adequately verbose
        if len(self.wavs)==1:
            for i in mimical_prior:
                if 'source' in i:
                    for j in mimical_prior[i]:
                        mimical_prior[i][j] = (mimical_prior[i][j],
                                               'Individual')
                else: mimical_prior[i] = (mimical_prior[i],
                                          'Individual')
        """

        # Set and run SourceExtractor if desired
        self.contmaps = np.ones_like(self.images)
        if se_maxdist == 'default':
            self.se_maxdist = 20
        else:
            self.se_maxdist = se_maxdist
        self.dilute = dilute
        self.dilute_radius = dilute_radius
        if self.se_clean:
            self.contmaps = create_contmaps(self.id, self.wavs, self.images,
                                            self.filter_names, self.contmaps,
                                            self.se_maxdist, self.dilute,
                                            self.dilute_radius, self.runtag)

        # Initiate the prior handler object, used to parse and translate priors
        self.phandler = priorHandler(mimical_prior, self.filter_names,
                                     self.wavs, self.images, self.runtag,
                                     self.id)
        self.sampler_prior_keys = self.phandler.keys
        print(f"Fitting -{self.phandler.nsources}- parameter submodels with" +
              f" -{self.phandler.nparam}- parameter Mimical fit with "
              f"dimensionality -{self.phandler.ndim}-.")

        # Get the keys of all Mimical properties per filter
        self.mimical_keys = []
        for key in self.mimical_prior.keys():
            if isinstance(self.mimical_prior[key], dict):
                for subkey in self.mimical_prior[key].keys():
                    self.mimical_keys.append(f"{key}:{subkey}")
            else:
                self.mimical_keys.append(key)

        # Code-timing interface
        self.calls = 0
        self.calltime = 0

    def get_residuals(self, param_vec):
        """ Calculate the residuals between the data and model. """

        # Translate unit cube prior sample into Mimical prior sample
        reverted = self.phandler.revert(param_vec)

        # Check if sampled model paramters are all within bounds
        voidcount = -1
        for key in self.mimical_prior.keys():
            if isinstance(self.mimical_prior[key], dict):
                for subkey in self.mimical_prior[key].keys():
                    voidcount += 1
                    bounds = self.mimical_prior[key][subkey][0]
                    if isinstance(bounds, tuple):
                        if (any(reverted[:, voidcount] < bounds[0])) | \
                           (any(reverted[:, voidcount] > bounds[1])):
                            return 'void', 'void'
                    else:
                        continue
            else:
                voidcount += 1
                bounds = self.mimical_prior[key][0]
                if isinstance(bounds, tuple):
                    if (any(reverted[:, voidcount] < bounds[0])) | \
                       (any(reverted[:, voidcount] > bounds[1])):
                        return 'void', 'void'
                else:
                    continue

        # Pull out model and mimical parameters
        modelpars = reverted[:, :np.sum(self.phandler.nsources)]
        psfarr = reverted[:, np.sum(self.phandler.nsources)]
        rmsarr = reverted[:, np.sum(self.phandler.nsources)+1]
        cpfarr = reverted[:, np.sum(self.phandler.nsources)+2]

        # If user provides RMS per pixel, override prior sample
        if isinstance(self.mimical_prior['rms'][0], list):
            if isinstance(self.mimical_prior['rms'][0][0], np.ndarray):
                rmsarr = np.array(self.mimical_prior['rms'][0])
        # If user wants mimical to infer RMS from image background, do so
        elif (isinstance(self.mimical_prior['rms'][0], str)):
            if (self.mimical_prior['rms'][0] == "Infer"):
                if not self.se_clean:
                    raise Exception("If using the 'Infer' special type for " +
                                    "RMS, must set se_clean=True.")

        # If user provides counts-per-flux per pixel, override prior sample
        if isinstance(self.mimical_prior['counts_per_flux'][0], list):
            if isinstance(self.mimical_prior['counts_per_flux'][0][0],
                          np.ndarray):
                cpfarr = np.array(self.mimical_prior['counts_per_flux'][0])

        # Update the model for sampled parameters
        newpars = torch.tensor(modelpars.astype(np.float32), device=self.accel)
        newpsfpas = torch.tensor(psfarr.astype(np.float32), device=self.accel)
        self.image_models.update_parameters(newpars, newpsfpas)

        # Update oversampling if 'auto' is chosen
        if isinstance(self.oversample, str):
            if self.oversample == 'auto':

                r_eff = modelpars[:, 1]
                n = modelpars[:, 2]
                coords = np.array((r_eff, n)).T
                oversamp_1 = self.interpolator_1(coords)
                oversamp_2 = self.interpolator_2(coords)
                oversamp_3 = self.interpolator_3(coords)
                oversampling = np.maximum(1,
                                          np.round(np.vstack([oversamp_1,
                                                              oversamp_2,
                                                              oversamp_3]).T))
                argmax = np.argmax(np.sum(oversampling, axis=1))

                autosamp = oversampling[argmax].astype(int).tolist()
                autorad = np.array(([1, max(2, r_eff[argmax]),
                                     max(3, 3*r_eff[argmax])])).tolist()
                self.image_models.update_oversampling(oversample=autosamp,
                                                      oversample_radii=autorad)

        # Discretize model to grid
        model = self.image_models.render().cpu().numpy()

        # If the model has NaNs, set to zero and blow up errors.
        if np.isnan(np.sum(model)):
            model = np.zeros_like(model)
            sigma = np.zeros_like(model) + 1e99
            print('Unphysical model detected.')
        # Calculate the error by the quadrature sum of rms and poisson errors
        else:
            sigma = np.sqrt(rmsarr.T**2 +
                            ((cpfarr.T**(-1/2))*np.sqrt(np.abs(model.T)))**2).T

        # Calculate the 3D mask
        contmask = self.contmaps == 1
        if rmsarr.shape == self.images.shape:
            contmask *= rmsarr != 0
        if cpfarr.shape == self.images.shape:
            contmask *= cpfarr != 0

        # Calculate the filter specific likelihood and add to total
        residuals = (self.images - model)[contmask]
        sigma = sigma[contmask]

        return residuals, sigma

    def lnlike(self, param_vec):
        """ Returns the log-likelihood for a given parameter vector. """

        t0 = time.time()

        residuals, sigma = self.get_residuals(param_vec)

        if isinstance(residuals, str):
            return -9.99e99

        norm = np.log((1/(np.sqrt(2*np.pi*(sigma**2)))))
        log_like_array = norm + ((-(residuals)**2) / (2*(sigma**2)))
        log_like = np.sum(log_like_array)

        self.calls += 1
        self.calltime += time.time()-t0

        return log_like

    def run(self, n_live=400, pool=None, oversample=None,
            oversample_boxlength=None, oversample_radii=None,
            gpu_acceleration=False, verbose_sampler=True):
        """ Run the sampler and save results.

        Parameters
        ----------

        n_live : int
            Number of live points in nested sampling algorithm.

        pool : none or int
            Number of cores to parallelise likelihood calculations to.

        oversample : int or list
            Oversample factor for the entire image or annuli defined by
            oversample radii.

        oversample_boxlength : int
            Width of box about image center to oversample within.

        oversample_radii : int or list
            Radii in which to oversample.

        gpu_acceleration : boolean
            Whether or not to accelerate the model generation onto the first
            available GPU, compatable with apple silicon.

        verbose : bool
            Whether or not to have verbose output from the Nautilus sampler.
        """

        # Set oversampling and find accelerator platform if available.
        self.oversample = oversample
        self.oversamp_bl = oversample_boxlength
        self.oversample_radii = oversample_radii
        if gpu_acceleration:
            curr_accel = torch.accelerator.current_accelerator()
            is_avail = torch.accelerator.is_available()
            self.accel = curr_accel if is_avail else torch.device('cpu')
        else:
            self.accel = torch.device('cpu')

        # Define dummy models in each filter which are updated during sampling
        if self.psfs is not None:
            psf_accel = torch.tensor(self.psfs.astype(np.float32),
                                     device=self.accel)
        else:
            psf_accel = self.psfs
        submodels = [self.submodel()]*len(self.phandler.nsources)
        self.image_models = ImageModel(torch.arange(self.images.shape[2],
                                                    device=self.accel),
                                       torch.arange(self.images.shape[1],
                                                    device=self.accel),
                                       submodels,
                                       psf=psf_accel,
                                       psf_pa=np.zeros(len(self.wavs)),
                                       oversample=self.oversample,
                                       oversample_boxlength=self.oversamp_bl,
                                       oversample_radii=self.oversample_radii)

        # Load automatic oversampling table if auto
        if isinstance(oversample, str):
            if oversample == 'auto':
                if not os.path.isfile(tabledir +
                                      '/table1_values.txt'):
                    make_oversampling_table()
                self.n_indices = np.loadtxt(tabledir + f'/n_values.txt')
                self.r_eff_indices = np.loadtxt(tabledir +
                                                f'/r_eff_values.txt')
                self.oversamp_tab = np.c_[
                    [np.loadtxt(tabledir +
                                f'/table{i+1}_values.txt') for i in range(3)]]
                self.interpolator_1 = RGI((self.r_eff_indices,
                                           self.n_indices),
                                          self.oversamp_tab[0],
                                          method='cubic')
                self.interpolator_2 = RGI((self.r_eff_indices,
                                           self.n_indices),
                                          self.oversamp_tab[1],
                                          method='cubic')
                self.interpolator_3 = RGI((self.r_eff_indices,
                                           self.n_indices),
                                          self.oversamp_tab[2],
                                          method='cubic')

        # Check if a posterior already exists for the object being fitted
        if os.path.isfile(dir_path+f'/mimical_output/posteriors{self.runtag}' +
                          f'/{self.id}_points.txt'):
            self.points = np.loadtxt(dir_path+f'/mimical_output/' +
                                     f'posteriors{self.runtag}/' +
                                     f'{self.id}_points.txt', dtype=np.float32)
            self.log_w = np.loadtxt(dir_path+f'/mimical_output/' +
                                    f'posteriors{self.runtag}/' +
                                    f'{self.id}_logw.txt', dtype=np.float32)
            self.log_l = np.loadtxt(dir_path+f'/mimical_output/' +
                                    f'posteriors{self.runtag}/' +
                                    f'{self.id}_logl.txt', dtype=np.float32)
            print(f"Loading existing posterior at " + dir_path +
                  f'/mimical_output/posteriors{self.runtag}')
            self.save_output()

        else:
            # Set the sampler prior
            t0 = time.time()
            sampler = Sampler(self.phandler.sampler_prior, self.lnlike,
                              n_live=n_live, pool=pool,
                              n_dim=self.phandler.ndim)
            sampler.run(verbose=verbose_sampler)
            print(f"Sampling time (minutes): {(time.time()-t0)/60}")
            self.points, self.log_w, self.log_l = sampler.posterior()
            self.save_output()

    def calc_chisq(self, param_vec):
        """ Calculates the Chisq value for the given parameter vector. """

        residuals, sigma = self.get_residuals(param_vec)
        chisq_arr = residuals**2 / sigma**2
        chisq = np.sum(chisq_arr)

        return chisq, len(chisq_arr)

    def save_output(self):
        """ Saves the percentiles of user parameters for each filter. """

        # Save the sampled points and corresponding log-weights
        np.savetxt(dir_path+f'/mimical_output/posteriors{self.runtag}/' +
                   f'{self.id}_points.txt', self.points)
        np.savetxt(dir_path+f'/mimical_output/posteriors{self.runtag}/' +
                   f'{self.id}_logw.txt', self.log_w)
        np.savetxt(dir_path+f'/mimical_output/posteriors{self.runtag}/' +
                   f'{self.id}_logl.txt', self.log_l)

        # Sample an appropriately weighted posterior for representative samples
        n_post = 10000
        print(self.points.shape[0])
        print(len(self.log_w))
        indices = np.random.choice(np.arange(self.points.shape[0]),
                                   size=n_post, p=np.exp(self.log_w))
        self.samples = self.points[indices]
        chisq, numd = self.calc_chisq(self.points[np.argmax(self.log_l)])

        # Save sampler values
        quan = np.percentile(self.samples, q=(16, 50, 84), axis=0)
        dic = {"id": self.id}
        for i in range(len(self.sampler_prior_keys)):
            key = list(self.sampler_prior_keys)[i]
            dic[key + "_16"] = [quan[0, i]]
            dic[key + "_50"] = [quan[1, i]]
            dic[key + "_84"] = [quan[2, i]]
        dic['chisq'] = chisq
        dic['numd'] = numd
        dic['red_chisq'] = chisq / (numd-self.phandler.ndim)
        df1 = pd.DataFrame(dic)

        # Save per filter values
        samp_filt = np.apply_along_axis(lambda p:
                                        self.phandler.revert(p).flatten(), 1,
                                        self.samples)
        samp_filt = samp_filt.reshape(self.samples.shape[0], len(self.wavs),
                                      np.sum(self.phandler.nsources)+3)
        quan = np.percentile(samp_filt, q=(16, 50, 84), axis=0)
        dic = {"id": self.id}
        for j in range(len(self.wavs)):
            for i in range(len(self.mimical_keys)):
                key = list(self.mimical_keys)[i]
                dic[key + "_" + self.filter_names[j] + "_16"] = [quan[0, j, i]]
                dic[key + "_" + self.filter_names[j] + "_50"] = [quan[1, j, i]]
                dic[key + "_" + self.filter_names[j] + "_84"] = [quan[2, j, i]]
        dic['chisq'] = chisq
        dic['numd'] = numd
        dic['red_chisq'] = chisq / (numd-self.phandler.ndim)
        df2 = pd.DataFrame(dic)

        # If not part of a catalogue fit, save individual
        if self.runtag == '':
            df1.to_csv(dir_path+f'/mimical_output/cats/{self.id}_'
                       'sampcat.csv', index=False)
            df2.to_csv(dir_path+f'/mimical_output/cats{self.runtag}_'
                       'filtcat.csv', index=False)

        # If part of a catalogue fit, append to it.
        else:
            if not os.path.isfile(dir_path + f'/mimical_output/' +
                                  f'cats{self.runtag}_sampcat.csv'):
                df1.to_csv(dir_path+f'/mimical_output/cats{self.runtag}'
                           '_sampcat.csv', index=False)
                df2.to_csv(dir_path+f'/mimical_output/cats{self.runtag}'
                           '_filtcat.csv', index=False)
            else:
                ridden1 = pd.read_csv(dir_path+f'/mimical_output/' +
                                      f'cats/{self.runtag}_sampcat.csv')
                ridden2 = pd.read_csv(dir_path+f'/mimical_output/' +
                                      f'cats/{self.runtag}_filtcat.csv')
                ridden1.index = ridden1['id'].values.astype('str')
                if self.id not in ridden1.index.values:
                    ridden1.loc[self.id] = df1.values[0]
                    ridden2.loc[self.id] = df2.values[0]
                    ridden1.to_csv(dir_path+f'/mimical_output/' +
                                   f'cats/{self.runtag}_sampcat.csv',
                                   index=False)
                    ridden2.to_csv(dir_path+f'/mimical_output/' +
                                   f'cats/{self.runtag}_filtcat.csv',
                                   index=False)
                else:
                    print('Object already written to catalogue.')

        # Save best model image for each filter
        param_vec = self.phandler.revert(self.points[np.argmax(self.log_l)])
        pars = param_vec[:, :np.sum(self.phandler.nsources)]
        psfarr = param_vec[:, np.sum(self.phandler.nsources)]
        self.image_models.update_parameters(torch.tensor(pars,
                                                         dtype=torch.float32,
                                                         device=self.accel),
                                            torch.tensor(psfarr,
                                                         dtype=torch.float32,
                                                         device=self.accel))
        best_models = self.image_models.render().cpu().numpy()
        for i in range(len(self.wavs)):
            np.savetxt(dir_path+f'/mimical_output/plots{self.runtag}/' +
                       f'{self.id}_best_model.txt', best_models[i])

        # Plot and save the corner plot
        mask = self.phandler.smask
        corner.corner(self.points.T[mask].T, weights=np.exp(self.log_w),
                      bins=20, labels=np.array(self.sampler_prior_keys)[mask],
                      color='black', plot_datapoints=False,
                      range=np.repeat(0.999, np.sum(mask)))
        plt.savefig(dir_path+f'/mimical_output/plots{self.runtag}/' +
                    f'{self.id}_corner.pdf', bbox_inches='tight')

        plt.close('all')

    def plot_model(self):
        """ Wrapper to plot output. """

        # Plot and save the maxL fit
        plotting.plot_best(self.images, self.wavs, self.image_models,
                           self.points[np.argmax(self.log_l)],
                           self.phandler, self.filter_names,
                           self.contmaps)
        plt.savefig(dir_path+f'/mimical_output/plots{self.runtag}/' +
                    f'{self.id}_fit_summary.pdf', bbox_inches='tight', dpi=500,
                    transparent=True)

        # Plot the trends with wavelength if multiband fit
        if len(self.wavs) > 1:
            plotting.plot_trends(self.wavs, self.samples, self.mimical_prior,
                                 self.phandler, self.mimical_keys)
            plt.savefig(dir_path+f'/mimical_output/plots{self.runtag}/' +
                        f'{self.id}_trends.pdf', bbox_inches='tight', dpi=500,
                        transparent=True)

        # Plot the errors used in fitting
        plotting.plot_errors(self.images, self.wavs, self.mimical_prior,
                             self.image_models,
                             self.points[np.argmax(self.log_l)],
                             self.phandler, self.filter_names,
                             self.contmaps)
        plt.savefig(dir_path+f'/mimical_output/plots{self.runtag}/' +
                    f'{self.id}_errors.pdf', bbox_inches='tight', dpi=500,
                    transparent=True)

        plt.close('all')
