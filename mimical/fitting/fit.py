import numpy as np
import matplotlib.pyplot as plt
import corner
from nautilus import Sampler
import time
import os
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator as RGI
from copy import deepcopy
from astropy.table import Table
from astropy.io import fits
import subprocess

from ..priors.prior_handler import priorHandler
from ..plotting import plotting
from ..utils import filter_set
from ..utils import get_segmaps
from ..utils import dilute_segmaps
from ..utils import make_oversampling_table
from ..models.submodels import Sersic
from ..models.imagemodel import ImageModel

dir_path = os.getcwd()
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
                 se_clean=False, se_maxdist=10,
                 dilute=True, dilute_radius=3, runtag='', rank=''):

        if not os.path.isdir(dir_path + f"/mimical_output/posteriors{runtag}"):
            subprocess.run(['mkdir', '-p',
                            dir_path+"/mimical_output/cats"])
            subprocess.run(['mkdir', '-p',
                            dir_path+f"/mimical_output/models{runtag}"])
            subprocess.run(['mkdir', '-p',
                            dir_path+f"/mimical_output/posteriors{runtag}"])

        # Set positional arguments
        self.id = id
        self.images = np.array((images))
        self.filt_list = filt_list
        if psfs is not None:
            self.psfs = np.array(([psfi/np.sum(psfi) for psfi in psfs]))
        else:
            self.psfs = psfs

        # Get the keys of all Mimical properties per filter
        self.mimical_prior = deepcopy(mimical_prior)
        self.mimical_keys = []
        self.submodels = []
        for key in mimical_prior.keys():
            if isinstance(mimical_prior[key], dict):
                for subkey in mimical_prior[key].keys():
                    if subkey == 'model':
                        self.submodels.append(mimical_prior[key][subkey])
                        del self.mimical_prior[key][subkey]
                    else:
                        self.mimical_keys.append(f"{key}:{subkey}")
            else:
                self.mimical_keys.append(key)

        # Set keyword arguments
        self.se_clean = se_clean
        self.runtag = runtag

        # Find the names and effective wavelengths of image filters
        self.filter_names = [x.split('/')[-1] for x in self.filt_list]
        filt_set = filter_set([dir_path+'/'+x for x in self.filt_list])
        self.wavs = (filt_set.eff_wavs / 1e4)

        # Set and run SourceExtractor if desired
        if se_clean:
            # 0=bg, 1=target, 2=contamination
            segmaps = get_segmaps(self.id,
                                  self.wavs,
                                  self.images,
                                  self.filter_names,
                                  se_maxdist,
                                  self.runtag)

            # 0 = sources, 1 = background
            self.bgmaps = [np.ones_like(sm)-(sm > 0) for sm in segmaps]
            if dilute:
                self.bgmaps = dilute_segmaps(self.bgmaps, dilute_radius)
            self.bgmaps = np.array((self.bgmaps))

            # 0=contamination, 1=target+background
            self.contmaps = [np.abs(sm - 2) for sm in segmaps]
            if dilute:
                contmaps_dil = dilute_segmaps(self.contmaps, dilute_radius)
                for i in range(len(self.contmaps)):
                    self.contmaps[i][contmaps_dil[i] == 0] = 0
                    self.contmaps[i][segmaps[i] == 1] = 1
                    self.contmaps[i][self.contmaps[i] != 0] = 1

            self.bgmaps = np.array((self.bgmaps))
            self.contmaps = np.array((self.contmaps))

        else:
            self.bgmaps = np.ones_like(self.images)
            self.contmaps = np.ones_like(self.images)

        # Initiate the prior handler object, used to parse and translate priors
        self.phandler = priorHandler(self.mimical_prior, self.filter_names,
                                     self.wavs, self.images, self.runtag,
                                     self.id, self.bgmaps)
        self.sampler_prior_keys = self.phandler.keys
        print(f"\nFitting object {self.id} with"
              f" -{self.phandler.nsources}- parameter submodels with"
              f" -{self.phandler.nparam}- parameter Mimical fit with "
              f"dimensionality -{self.phandler.ndim}-.")

        # Code-timing interface
        self.calls = 0
        self.calltime = 0

        self.rank = rank

    @torch.inference_mode()
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
                            print('Unphysical model detected.')
                            return 'void', 'void'
                    else:
                        continue
            else:
                voidcount += 1
                bounds = self.mimical_prior[key][0]
                if isinstance(bounds, tuple):
                    if (any(reverted[:, voidcount] < bounds[0])) | \
                       (any(reverted[:, voidcount] > bounds[1])):
                        print('Unphysical model detected.')
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
                autosamp, autorad = self.automatic_oversampling(modelpars)
                self.image_models.update_oversampling(oversample=autosamp,
                                                      oversample_radii=autorad)

        # Discretize model to grid
        model = self.image_models.render().cpu().numpy()

        # If the model has NaNs, set to zero and blow up errors.
        if np.isnan(np.sum(model)):
            return 'void', 'void'

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

        t1 = time.time()
        residuals, sigma = self.get_residuals(param_vec)

        if isinstance(residuals, str):
            return -1e99

        norm = np.log((1/(np.sqrt(2*np.pi*(sigma**2)))))
        log_like_array = norm + ((-(residuals)**2) / (2*(sigma**2)))
        log_like = np.sum(log_like_array)

        self.calls += 1
        self.calltime += time.time()-t1

        '''
        if self.calls % 100 == 0:
            print(f"Average call time at {self.calls}:"
                  f"{self.calltime/self.calls}")
        '''

        return log_like

    def run(self, n_live=1000, pool=None, oversample=None,
            oversample_bl=None, oversample_radii=None,
            gpu_acceleration=False, verbose_sampler=True,
            timeout=np.inf, rotmethod='interpolation'):
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

        oversample_bl : int
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
        self.oversample_bl = oversample_bl
        self.oversample_radii = oversample_radii
        if gpu_acceleration:
            curr_accel = torch.accelerator.current_accelerator()
            is_avail = torch.accelerator.is_available()
            if is_avail:
                print(f"Successfully found GPU at '{curr_accel}'.")
                self.accel = curr_accel
            else:
                print(f"Failed to find GPU, using CPU instead.")
                self.accel = torch.device('cpu')
        else:
            self.accel = torch.device('cpu')

        # Define dummy models in each filter which are updated during sampling
        if self.psfs is not None:
            psf_accel = torch.tensor(self.psfs.astype(np.float32),
                                     device=self.accel)
        else:
            psf_accel = self.psfs

        self.image_models = ImageModel(torch.arange(self.images.shape[2],
                                                    device=self.accel),
                                       torch.arange(self.images.shape[1],
                                                    device=self.accel),
                                       self.submodels,
                                       psf=psf_accel,
                                       psf_pa=np.zeros(len(self.wavs)),
                                       oversample=self.oversample,
                                       oversample_bl=self.oversample_bl,
                                       oversample_radii=self.oversample_radii,
                                       rotmethod=rotmethod)

        # Load automatic oversampling table if auto
        if isinstance(self.oversample, str):
            if self.oversample == 'auto':
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
        if os.path.isfile(dir_path+f'/mimical_output/posteriors{self.runtag}'
                          f'/{self.id}.fits'):
            posterior = Table.read(dir_path+f'/mimical_output/posteriors'
                                   f'{self.runtag}/{self.id}.fits')
            print(f"Loading existing posterior for object {self.id} at "
                  + dir_path +
                  f'/mimical_output/posteriors{self.runtag}/{self.id}.fits')
            posterior = np.column_stack([posterior[col] for
                                         col in posterior.colnames]
                                        ).astype(np.float32)
            self.samples = posterior[:, :-2]
            self.log_l = posterior[:, -2]
            self.success = bool(posterior[:, -1][0])
            self.save_output()

        else:
            # Set the sampler prior
            sampler = Sampler(self.phandler.sampler_prior, self.lnlike,
                              n_live=n_live, pool=pool,
                              n_dim=self.phandler.ndim)
            t0 = time.time()
            self.success = sampler.run(verbose=verbose_sampler,
                                       timeout=timeout * 60)
            sampling_time = (time.time()-t0)/60
            print(f"Sampling time for object {self.id} (minutes):"
                  f" {sampling_time}")

            raw_points, raw_log_w, raw_log_l = sampler.posterior()
            n_post = 10000
            indices = np.random.choice(np.arange(raw_points.shape[0]),
                                       size=n_post, p=np.exp(raw_log_w))
            self.samples = raw_points[indices]
            self.log_l = raw_log_l[indices]
            self.save_output()

    def calc_chisq(self, param_vec):
        """ Calculates the Chisq value for the given parameter vector. """

        residuals, sigma = self.get_residuals(param_vec)
        chisq_arr = residuals**2 / sigma**2
        chisq = np.sum(chisq_arr)

        return chisq, len(chisq_arr)

    def automatic_oversampling(self, modelpars):
        """ Function to determine bets oversampling properties
        for Sersic fits. """

        if not isinstance(self.submodels[0], Sersic):
            raise Exception("To use automatic oversampling,"
                            " the primary source should be a"
                            "a Sersic profile.")

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

        return autosamp, autorad

    def save_output(self):
        """ Saves the percentiles of user parameters for each filter. """

        # Save the sampled points and corresponding log-weights
        posterior = np.c_[self.samples, self.log_l,
                          [int(self.success)]*len(self.log_l)]
        df = pd.DataFrame(posterior,
                          columns=[*self.sampler_prior_keys,
                                   'logL', 'success'])
        Table.from_pandas(df).write(dir_path + f'/mimical_output/posteriors'
                                    f'{self.runtag}/{self.id}.fits',
                                    overwrite=True)

        self.maxL_sample = self.samples[np.argmax(self.log_l)]
        chisq, numd = self.calc_chisq(self.maxL_sample)

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
        dic['success'] = self.success
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
        dic['success'] = self.success
        df2 = pd.DataFrame(dic)

        # If not part of a catalogue fit, save individual
        if self.runtag == '':
            df1.to_csv(dir_path+f'/mimical_output/cats/{self.id}.csv',
                       index=False)
            if len(self.wavs) > 1:
                df2.to_csv(dir_path+'/mimical_output/cats/'
                           f'{self.id}_perfilter.csv', index=False)

        # If part of a catalogue fit, append to it.
        else:
            if not os.path.isfile(dir_path + f'/mimical_output/' +
                                  f'cats{self.runtag}{self.rank}.csv'):
                df1.to_csv(dir_path+f'/mimical_output/cats{self.runtag}'
                           f'{self.rank}.csv', index=False)
                if len(self.wavs) > 1:
                    df2.to_csv(dir_path+f'/mimical_output/cats{self.runtag}'
                               f'{self.rank}_perfilter.csv', index=False)
            else:
                ridden1 = pd.read_csv(dir_path+f'/mimical_output/' +
                                      f'cats/{self.runtag}{self.rank}.csv')
                if len(self.wavs) > 1:
                    ridden2 = pd.read_csv(dir_path+f'/mimical_output/'
                                          f'cats/{self.runtag}'
                                          f'{self.rank}_perfilter.csv')
                ridden1.index = ridden1['id'].values.astype('str')
                if self.id not in ridden1.index.values:
                    ridden1.loc[self.id] = df1.values[0]
                    if len(self.wavs) > 1:
                        ridden2.loc[self.id] = df2.values[0]
                    ridden1.to_csv(dir_path+f'/mimical_output/' +
                                   f'cats/{self.runtag}{self.rank}.csv',
                                   index=False)
                    if len(self.wavs) > 1:
                        ridden2.to_csv(dir_path+f'/mimical_output/'
                                       f'cats/{self.runtag}'
                                       f'{self.rank}_perfilter.csv',
                                       index=False)

        # Save best model image for each filter
        param_vec = self.phandler.revert(self.maxL_sample)
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
            hdu = fits.PrimaryHDU(best_models[i])
            hdu.writeto(dir_path+f'/mimical_output/models{self.runtag}/'
                        f'{self.id}_best_model.fits', overwrite=True)

        print(f"Object {self.id} done.")

    def save_plots(self):
        """ Wrapper to plot output. """

        if not os.path.isdir(dir_path + f"/mimical_output/plots{self.runtag}"):
            subprocess.run(['mkdir', '-p', dir_path + f"/mimical_output/"
                            f"plots{self.runtag}/summary_plots"])
            subprocess.run(['mkdir', '-p', dir_path + f"/mimical_output/"
                            f"plots{self.runtag}/corner_plots"])
            subprocess.run(['mkdir', '-p', dir_path + f"/mimical_output/"
                            f"plots{self.runtag}/error_plots"])
            subprocess.run(['mkdir', '-p', dir_path + f"/mimical_output/"
                            f"plots{self.runtag}/trend_plots"])

        # Plot and save the corner plot
        mask = self.phandler.smask
        if self.success:
            range = np.repeat(0.999, np.sum(mask))
            corner.corner(self.samples.T[mask].T,
                          bins=20,
                          labels=np.array(self.sampler_prior_keys)[mask],
                          color='black',
                          plot_datapoints=False,
                          range=range)
            plt.savefig(dir_path+f'/mimical_output/plots{self.runtag}/'
                        f'corner_plots/{self.id}_corner.pdf',
                        bbox_inches='tight',
                        transparent=True)

        # Plot and save the maxL fit
        if isinstance(self.oversample, str):
            if self.oversample == 'auto':
                modelpars = self.phandler.revert(self.maxL_sample)
                res = self.automatic_oversampling(modelpars)
                oversample, oversample_radii = res
        else:
            oversample = self.oversample
            oversample_radii = self.oversample_radii

        plotting.plot_best(self.images, self.wavs, self.image_models,
                           self.maxL_sample,
                           self.phandler, self.filter_names,
                           self.contmaps, oversample,
                           self.oversample_bl, oversample_radii)
        plt.savefig(dir_path+f'/mimical_output/plots{self.runtag}/'
                    f'summary_plots/{self.id}_fit_summary.pdf',
                    bbox_inches='tight', dpi=300,
                    transparent=True)

        # Plot the trends with wavelength if multiband fit
        if len(self.wavs) > 1:
            plotting.plot_trends(self.wavs, self.samples, self.mimical_prior,
                                 self.phandler, self.mimical_keys)
            plt.savefig(dir_path+f'/mimical_output/plots{self.runtag}/'
                        f'trend_plots/{self.id}_trends.pdf',
                        bbox_inches='tight',
                        transparent=True)

        # Plot the errors used in fitting
        plotting.plot_errors(self.images, self.wavs, self.mimical_prior,
                             self.image_models,
                             self.maxL_sample,
                             self.phandler, self.filter_names,
                             self.contmaps)
        plt.savefig(dir_path+f'/mimical_output/plots{self.runtag}/'
                    f'error_plots/{self.id}_errors.pdf',
                    bbox_inches='tight', dpi=300,
                    transparent=True)

        plt.close('all')
