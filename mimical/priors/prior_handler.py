import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits

from .prior_types import individual, polynomial, powerlaw

dir_path = os.getcwd()


class priorHandler(object):
    """ Contains the functionality for translating Mimical priors into  sampler
    priors, and translating sampler samples into model parameters in each
    filter.

    Parameters
    ----------

    mimical_prior : dict
        The user specified prior which set out the priors for the model
        parameters and passes information about whether to let these vary
        for each filter or whether they follow an order-specified polynomial
        relationship.

    filter_names : list of str
        A list of filter names e.g., ['F356W', 'F444W', ...]

    wavs : 1darray
        A 1D array of effective wavelengths corresponding to each filter.

    images : 3darray
        3D image with slices for each filter.

    runtag : str
        A name for the mimical catalogue run.

    id : str
        An ID for the fitting run. Only really used for output files.
    """

    def __init__(self, mimical_prior, filter_names, wavs,
                 images, runtag, id, bgmaps=None):
        self.mimical_prior = mimical_prior
        self.mimical_keys = []
        for key in self.mimical_prior.keys():
            if isinstance(self.mimical_prior[key], dict):
                for subkey in self.mimical_prior[key].keys():
                    self.mimical_keys.append(f"{key}:{subkey}")
            else:
                self.mimical_keys.append(key)

        self.filter_names = filter_names
        self.wavs = wavs
        self.nsources, self.nparam, self.ndim, \
            self.keys, self.smask = self.calculate_dimensionality()

        self.images = images
        self.runtag = runtag
        self.id = id

        if isinstance(self.mimical_prior['rms'][0], str):
            if self.mimical_prior['rms'][0] == "Infer":
                self.rms = []
                for i in range(len(self.wavs)):
                    bg = self.images[i][bgmaps[i] == 1]
                    rmsi = np.sqrt(np.mean(np.square(bg)))
                    self.rms.append(rmsi)

            else:
                raise Exception("Must run Sextractor if using type "
                                "'Infer'.")

    def sampler_prior(self, x):
        """ Defines the prior used for sampling. Transforms the unit cube. """

        # Create empty mimical parameter array
        theta = np.zeros(self.nparam)
        # Keep record of the current element in the unit cube
        xcount = 0
        # Keep record of the current element in the mimical parameter array
        thetac = 0

        # Loop over Mimical parameters
        for key in self.mimical_prior.keys():

            if "source" in key:

                sourcedic = self.mimical_prior[key]

                for sourcekey in sourcedic.keys():

                    # Load in the Mimical prior element
                    param_prior_traits = sourcedic[sourcekey]
                    prior_dist = param_prior_traits[0]

                    # For fitted params
                    if isinstance(prior_dist, tuple):

                        param_fit_type = param_prior_traits[1]

                        # If user specifies 'Individual', add a free parameter
                        # for each filter.
                        if param_fit_type == "Individual":
                            indysamps = individual(x[xcount:
                                                     xcount+len(self.wavs)],
                                                   prior_dist)
                            theta[thetac:thetac+len(self.wavs)] = indysamps
                            thetac += len(self.wavs)
                            xcount += len(self.wavs)

                        # If user specifies 'Polynomial', add a free parameter
                        # for each polynomial coefficient.
                        elif param_fit_type == "Polynomial":
                            poly_order = param_prior_traits[2]
                            polysamps = polynomial(x[xcount:
                                                     xcount+poly_order+1],
                                                   prior_dist, poly_order,
                                                   self.wavs)
                            theta[thetac:thetac+poly_order+1] = polysamps
                            thetac += poly_order+1
                            xcount += poly_order+1

                        # If user specifies 'Power-law', add a free parameter
                        # for each power law coefficient.
                        elif param_fit_type == "Power-law":
                            powerbounds = param_prior_traits[2]
                            epsilon = param_prior_traits[3]
                            plaw_samps = powerlaw(x[xcount:xcount+3],
                                                  prior_dist, self.wavs,
                                                  powerbounds, epsilon)
                            theta[thetac:thetac+3] = plaw_samps
                            thetac += 3
                            xcount += 3

                        else:
                            raise Exception("Fitting type not supported, "
                                            "please choose either "
                                            "'Individual', 'Polynomial' "
                                            "or 'Power-law'.")

                    # For fixed params
                    elif isinstance(prior_dist,
                                    (float, int, list, np.ndarray)):

                        param_fit_type = param_prior_traits[1]

                        # If fixed for each individual filter, set for each
                        if (param_fit_type == "Individual"):

                            if isinstance(prior_dist, (float, int)):
                                theta[thetac:
                                      thetac+len(self.wavs)] = prior_dist
                                thetac += len(self.wavs)

                            elif isinstance(prior_dist, list):
                                if not isinstance(prior_dist[0], np.ndarray):
                                    theta[thetac:
                                          thetac+len(self.wavs)] = prior_dist
                                    thetac += len(self.wavs)
                                # If user supplies values for each image pixel
                                # (pertinent for RMS etc.), then pass the mean
                                # to the prior samples. This is required for
                                # generality but is overwritten later in the
                                # likelihood function.
                                else:
                                    meaner = np.mean(np.array((prior_dist)),
                                                     axis=(1, 2))
                                    theta[thetac:
                                          thetac+len(self.wavs)] = meaner
                                    thetac += len(self.wavs)

                            else:
                                raise Exception('Must pass float/int/list for '
                                                'a multiband fit. The list can'
                                                ' be a list of floats/ints or '
                                                'a list of arrays.')

                        # If user supplies polynomial coefficients, set them.
                        elif param_fit_type == "Polynomial":
                            poly_order = param_prior_traits[2]
                            theta[thetac:
                                  thetac+(poly_order+1)] = prior_dist
                            thetac += (poly_order+1)

                        # If user supplies power-law coefficients, set them.
                        elif param_fit_type == "Power-law":
                            theta[thetac:
                                  thetac+3] = prior_dist
                            thetac += 3

                        else:
                            raise Exception("Fitting type not supported, "
                                            "please choose either "
                                            "'Individual', 'Polynomial' "
                                            "or 'Power-law'.")

            elif (('psf_pa' in key) |
                  ('rms' in key) |
                  ('counts_per_flux' in key)):

                # Load in the Mimical prior element
                param_prior_traits = self.mimical_prior[key]
                prior_dist = param_prior_traits[0]

                # For fitted params
                if isinstance(prior_dist, tuple):

                    param_fit_type = param_prior_traits[1]

                    # If user specifies 'Individual', add a free parameter for
                    # each filter.
                    if param_fit_type == "Individual":
                        indysamp = individual(x[xcount:xcount+len(self.wavs)],
                                              prior_dist)
                        theta[thetac:thetac+len(self.wavs)] = indysamp
                        thetac += len(self.wavs)
                        xcount += len(self.wavs)

                    # If user specifies 'Polynomial', add a free parameter for
                    # each polynomial coefficient.
                    elif param_fit_type == "Polynomial":
                        poly_order = param_prior_traits[2]
                        polysamp = polynomial(x[xcount:xcount+poly_order+1],
                                              prior_dist, poly_order,
                                              self.wavs)
                        theta[thetac:thetac+poly_order+1] = polysamp
                        thetac += poly_order+1
                        xcount += poly_order+1

                    # If user specifies 'Power-law', add three free parameters.
                    elif param_fit_type == "Power-law":
                        powerbounds = param_prior_traits[2]
                        epsilon = param_prior_traits[3]
                        theta[thetac:
                              thetac+3] = powerlaw(x[xcount:xcount+3],
                                                   prior_dist, self.wavs,
                                                   powerbounds, epsilon)
                        thetac += 3
                        xcount += 3

                    else:
                        raise Exception("Fitting type not supported, please "
                                        "choose either 'Individual', "
                                        "'Polynomial' or 'Power-law'.")

                # For fixed params
                elif isinstance(prior_dist, (float, int, list, np.ndarray)):

                    param_fit_type = param_prior_traits[1]

                    # If fixed for each individual filter, set for each
                    if (param_fit_type == "Individual"):

                        if isinstance(prior_dist, (float, int)):
                            theta[thetac:
                                  thetac+len(self.wavs)] = prior_dist
                            thetac += len(self.wavs)

                        elif isinstance(prior_dist, list):
                            if not isinstance(prior_dist[0], np.ndarray):
                                theta[thetac:
                                      thetac+len(self.wavs)] = prior_dist
                                thetac += len(self.wavs)
                            # If user supplies values for each image pixel
                            # (pertinent for RMS etc.), then pass the mean
                            # to the prior samples. This is required for
                            # generality but is overwritten later in the
                            # likelihood function.
                            else:
                                meaner = np.mean(np.array((prior_dist)),
                                                 axis=(1, 2))
                                theta[thetac:
                                      thetac+len(self.wavs)] = meaner
                                thetac += len(self.wavs)

                        else:
                            raise Exception('Must pass float/int/list for a '
                                            'multiband fit. The list can be a '
                                            'list of floats/ints or a list of '
                                            'arrays.')

                    # If user supplies polynomial coefficients, set them.
                    elif param_fit_type == "Polynomial":
                        poly_order = param_prior_traits[2]
                        theta[thetac:
                              thetac+(poly_order+1)] = prior_dist
                        thetac += (poly_order+1)

                    # If user supplies power-law coefficients, set them.
                    elif param_fit_type == "Power-law":
                        theta[thetac:thetac+3] = prior_dist
                        thetac += 3

                    else:
                        raise Exception("Fitting type not supported, please "
                                        "choose either 'Individual', "
                                        "'Polynomial' or 'Power-law'.")

                # For inferred RMS
                elif (key == 'rms') & (isinstance(prior_dist, str)):
                    if (prior_dist == 'Infer'):
                        theta[thetac:thetac+len(self.wavs)] = self.rms
                        thetac += len(self.wavs)
                    else:
                        raise Exception("The only special prior type for "
                                        "RMS is 'Infer'.")

            # For wrongly inputted prior types
            else:
                raise Exception("Mimical only accepts a min/max tuple for "
                                "fitting, or a list/ndarray/float/int for "
                                "fixing.")

        return theta

    def revert(self, param_dict):
        """ Translate a sampler sample into a sample of model parameters for
        each filter."""

        # Empty parameter array
        params_final = np.zeros((len(self.wavs), np.sum(self.nsources)+3))
        ind = 0
        count = 0

        # Loop over model parameters
        keys = list(self.mimical_prior.keys())
        for i in range(len(keys)):

            if "source" in keys[i]:

                sourcedic = self.mimical_prior[keys[i]]

                for sourcekey in sourcedic.keys():

                    # Load in the Mimical prior element
                    param_prior_traits = sourcedic[sourcekey]
                    prior_dist = param_prior_traits[0]
                    param_fit_type = param_prior_traits[1]

                    # If individual, add the sample for each filter
                    if param_fit_type == "Individual":
                        params_final[:, ind] = param_dict[count:
                                                          count+len(self.wavs)]
                        ind += 1
                        count += len(self.wavs)

                    # If polynomial, calculate the expected parameter in each
                    # filter given its effective wavlength
                    elif param_fit_type == "Polynomial":
                        poly_order = param_prior_traits[2]
                        coeffs = param_dict[count:count+poly_order+1]
                        tiler = np.tile(self.wavs-self.wavs[0],
                                        (poly_order+1, 1)).T
                        polywavs = np.pow(tiler, np.arange(poly_order+1))
                        comps = coeffs * polywavs
                        comps_summed = np.sum(comps, axis=1)
                        params_final[:, ind] = comps_summed
                        ind += 1
                        count += poly_order+1

                    # If power-law, calculate the expected parameter in each
                    # filter given its effective wavlength
                    elif param_fit_type == "Power-law":
                        epsilon = param_prior_traits[3]
                        coeffs = param_dict[count:count+3]
                        tiler = np.tile(((self.wavs-self.wavs[0])+epsilon) /
                                        ((self.wavs[-1]-self.wavs[0])+epsilon),
                                        (2, 1)).T
                        polywavs = np.pow(tiler, np.array([0, coeffs[2]]))
                        comps = np.array([coeffs[0],
                                          coeffs[1]-coeffs[0]]) * polywavs
                        comps_summed = np.sum(comps, axis=1)
                        params_final[:, ind] = comps_summed
                        ind += 1
                        count += 3

                    else:
                        raise Exception("Fitting type not supported, please "
                                        "choose either 'Individual', "
                                        "'Polynomial' or 'Power-law'.")

            elif (('psf_pa' in keys[i]) |
                  ('rms' in keys[i]) |
                  ('counts_per_flux' in keys[i])):

                # Load in the Mimical prior element
                param_prior_traits = self.mimical_prior[keys[i]]
                prior_dist = param_prior_traits[0]

                # If using 'Infer' special type for the RMS parameter
                if (keys[i] == 'rms') & (isinstance(prior_dist, str)):
                    if (prior_dist == 'Infer'):
                        params_final[:, ind] = param_dict[count:
                                                          count+len(self.wavs)]
                        ind += 1
                        count += len(self.wavs)
                    else:
                        raise Exception(' ')

                else:
                    param_fit_type = param_prior_traits[1]

                    # If individual, add the sample for each filter
                    if param_fit_type == "Individual":
                        params_final[:, ind] = param_dict[count:
                                                          count+len(self.wavs)]
                        ind += 1
                        count += len(self.wavs)

                    # If polynomial, calculate the expected parameter in each
                    # filter given its effective wavlength
                    elif param_fit_type == "Polynomial":
                        poly_order = param_prior_traits[2]
                        coeffs = param_dict[count:count+poly_order+1]
                        polywavs = np.pow(np.tile(self.wavs-self.wavs[0],
                                                  (poly_order+1, 1)).T,
                                          np.arange(poly_order+1))
                        comps = coeffs * polywavs
                        comps_summed = np.sum(comps, axis=1)
                        params_final[:, ind] = comps_summed
                        ind += 1
                        count += poly_order+1

                    # If power-law, calculate the expected parameter in each
                    # filter given its effective wavlength.
                    elif param_fit_type == "Power-law":
                        epsilon = param_prior_traits[3]
                        coeffs = param_dict[count:count+3]
                        tiler = np.tile(((self.wavs-self.wavs[0])+epsilon) /
                                        ((self.wavs[-1]-self.wavs[0])+epsilon),
                                        (2, 1)).T
                        polywavs = np.pow(tiler, np.array([0, coeffs[2]]))
                        comps = np.array([coeffs[0],
                                          coeffs[1]-coeffs[0]]) * polywavs
                        comps_summed = np.sum(comps, axis=1)
                        params_final[:, ind] = comps_summed
                        ind += 1
                        count += 3

                    else:
                        raise Exception("Fitting type not supported, please "
                                        "choose either 'Individual', "
                                        "'Polynomial' or 'Power-law'.")

        return params_final

    def calculate_dimensionality(self):
        """ Calculates the model parameters, Mimical parameters and
        dimensionality of the sampling algorithm. """

        keys = []
        nsources = []
        nparam = 0
        ndim = 0
        smask = []

        # Loop over model parameters
        sourcecount = 0
        for key in self.mimical_prior.keys():

            if "source" in key:

                sourcedic = self.mimical_prior[key]
                sourcecount += 1
                nsources.append(0)

                for sourcekey in sourcedic.keys():
                    nsources[sourcecount-1] += 1

                    # Load in the Mimical prior element
                    param_prior_traits = sourcedic[sourcekey]
                    prior_dist = param_prior_traits[0]

                    # For fitted params
                    if isinstance(prior_dist, tuple):

                        param_fit_type = param_prior_traits[1]

                        if param_fit_type == "Individual":
                            for i in range(len(self.wavs)):
                                keys.append(f'{key}:{sourcekey}_'
                                            f'{self.filter_names[i]}')
                                smask.append(True)
                                nparam += 1
                                ndim += 1

                        elif param_fit_type == "Polynomial":
                            poly_order = param_prior_traits[2]
                            for i in range(0, poly_order+1):
                                keys.append(f'{key}:{sourcekey}_P{i}')
                                smask.append(True)
                                nparam += 1
                                ndim += 1

                        elif param_fit_type == "Power-law":
                            for i in range(3):
                                keys.append(f'{key}:{sourcekey}_PL{i}')
                                smask.append(True)
                                nparam += 1
                                ndim += 1

                        else:
                            raise Exception("Fitting type not supported, "
                                            "please choose either "
                                            "'Individual', 'Polynomial'"
                                            " or 'Power-law'.")
                    # For fixed params
                    elif isinstance(prior_dist, (float,
                                                 int,
                                                 list,
                                                 np.ndarray)):

                        param_fit_type = param_prior_traits[1]

                        if param_fit_type == "Individual":
                            for i in range(len(self.wavs)):
                                keys.append(f'{key}:{sourcekey}_'
                                            f'{self.filter_names[i]}')
                                smask.append(False)
                                nparam += 1

                        elif param_fit_type == "Polynomial":
                            poly_order = param_prior_traits[2]
                            for i in range(0, poly_order+1):
                                keys.append(f'{key}:{sourcekey}_P{i}')
                                smask.append(False)
                                nparam += 1

                        elif param_fit_type == "Power-law":
                            for i in range(3):
                                keys.append(f'{key}:{sourcekey}_PL{i}')
                                smask.append(False)
                                nparam += 1

                        else:
                            raise Exception("Fitting type not supported, "
                                            "please choose either "
                                            "'Individual', 'Polynomial'"
                                            " or 'Power-law'.")

            elif (('psf_pa' in key) |
                  ('rms' in key) |
                  ('counts_per_flux' in key)):

                # Load in the Mimical prior element
                param_prior_traits = self.mimical_prior[key]
                prior_dist = param_prior_traits[0]

                # For fitted params
                if isinstance(prior_dist, tuple):

                    param_fit_type = param_prior_traits[1]

                    if param_fit_type == "Individual":
                        for i in range(len(self.wavs)):
                            keys.append(f'{key}_{self.filter_names[i]}')
                            smask.append(True)
                            nparam += 1
                            ndim += 1

                    elif param_fit_type == "Polynomial":
                        poly_order = param_prior_traits[2]
                        for i in range(0, poly_order+1):
                            keys.append(key+f'_C{i}')
                            smask.append(True)
                            nparam += 1
                            ndim += 1

                    elif param_fit_type == "Power-law":
                        for i in range(3):
                            keys.append(key+f'_P{i}')
                            smask.append(True)
                            nparam += 1
                            ndim += 1

                    else:
                        raise Exception("Fitting type not supported, please "
                                        "choose either 'Individual', "
                                        "'Polynomial' or 'Power-law'.")
                # For fixed params
                elif isinstance(prior_dist, (float,
                                             int,
                                             list,
                                             np.ndarray)):

                    param_fit_type = param_prior_traits[1]

                    if param_fit_type == "Individual":
                        for i in range(len(self.wavs)):
                            keys.append(f'{key}_{self.filter_names[i]}')
                            smask.append(False)
                            nparam += 1

                    elif param_fit_type == "Polynomial":
                        poly_order = param_prior_traits[2]
                        for i in range(0, poly_order+1):
                            keys.append(key+f'_C{i}')
                            smask.append(False)
                            nparam += 1

                    elif param_fit_type == "Power-law":
                        for i in range(3):
                            keys.append(key+f'_P{i}')
                            smask.append(False)
                            nparam += 1

                    else:
                        raise Exception("Fitting type not supported, please "
                                        "choose either 'Individual', "
                                        "'Polynomial' or 'Power-law'.")

                # For inferred RMS
                elif (key == 'rms') & (isinstance(prior_dist, str)):
                    if (prior_dist == 'Infer'):
                        for i in range(len(self.wavs)):
                            keys.append(f'{key}_{self.filter_names[i]}')
                            smask.append(False)
                            nparam += 1
                    else:
                        raise Exception('')

        return nsources, nparam, ndim, keys, smask

    def check_priors(self, n, type='sampler'):
        """ Sample the fitted prior volume n times."""

        unit_cube = np.random.rand(n, self.nparam)

        if type == 'sampler':
            samples_sampler = np.apply_along_axis(self.sampler_prior, 1,
                                                  unit_cube)
            return samples_sampler, self.keys

        elif type == 'mimical':
            samp_filt = np.apply_along_axis(lambda uv: self.revert(
                self.sampler_prior(uv)).flatten(), 1, unit_cube)

            keys = []
            for j in range(len(self.filter_names)):
                for i in range(len(self.mimical_keys)):
                    key = self.mimical_keys[i]
                    keys.append(f"{key}_{self.filter_names[j]}")

            return samp_filt, keys

        else:
            raise Exception("'type' must be either 'sampler' or 'mimical'.")

    def check_physical(self, samp_filt):
        """ Check what fraction of Mimical prior samples are physical. """

        n = len(samp_filt)
        mask = np.arange(n) == np.arange(n)

        for i in range(n):

            sampshape = (len(self.wavs), len(list(self.mimical_keys)))
            samples_now = samp_filt[i].reshape(*sampshape)

            # Check if sampled model paramters are all within bounds
            voidcount = -1
            for key in self.mimical_prior.keys():
                if isinstance(self.mimical_prior[key], dict):
                    for subkey in self.mimical_prior[key].keys():
                        voidcount += 1
                        bounds = self.mimical_prior[key][subkey][0]
                        if isinstance(bounds, tuple):
                            if ((any(samples_now[:, voidcount] <
                                     bounds[0])) |
                                (any(samples_now[:, voidcount] >
                                     bounds[1]))):
                                mask[i] = False

                        else:
                            continue
                else:
                    voidcount += 1
                    bounds = self.mimical_prior[key][0]
                    if isinstance(bounds, tuple):
                        if ((any(samples_now[:, voidcount] <
                                 bounds[0])) |
                            (any(samples_now[:, voidcount] >
                                 bounds[1]))):
                            mask[i] = False
                    else:
                        continue

        print(np.sum(mask)/len(mask))
        return mask

    def plot_samples(self, n, key):
        """ Plot prior samples. """

        samp_filt, keys = self.check_priors(n=n, type='mimical')
        mask = self.check_physical(samp_filt)
        physical = samp_filt[mask]

        fig, ax = plt.subplots()

        for i in range(len(physical)):
            curr_sample = physical[i].reshape(len(self.wavs),
                                              len(self.mimical_keys))
            ofinterest = curr_sample[:, list(self.mimical_keys).index(key)]
            ax.plot(self.wavs, ofinterest, color='black', alpha=1)
        ax.set_ylabel(key)
        ax.set_xlabel('$\\lambda$')

        return fig, ax
