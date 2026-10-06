import numpy as np
import matplotlib.pyplot as plt
import os

from .prior_relationships import polynomial, powerlaw

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

    def __init__(self, mimical_prior, mimical_keys, filter_names, wavs,
                 images, runtag, id):
        self.mimical_prior = mimical_prior
        self.mimical_keys = mimical_keys
        self.filter_names = filter_names
        self.wavs = wavs
        self.nsources, self.nparam, self.ndim, \
            self.keys, self.smask = self.calculate_dimensionality()
        self.images = images
        self.runtag = runtag
        self.id = id

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
                    param_prior = sourcedic[sourcekey]

                    # If same prior for each image
                    if ((callable(param_prior) |
                         isinstance(param_prior,
                                    (int, float, np.ndarray)))):
                        # Fitted
                        if callable(param_prior):
                            theta[thetac] = param_prior(x[xcount])
                            xcount += 1
                            thetac += 1
                        # Fixed
                        else:
                            if isinstance(param_prior, (int, float)):
                                theta[thetac] = param_prior
                            elif isinstance(param_prior, (np.ndarray)):
                                theta[thetac] = np.mean(param_prior)
                            thetac += 1

                    # If different prior for each image
                    elif (isinstance(param_prior, (tuple, list)) &
                          (not isinstance(param_prior[1], str))):
                        if ((callable(param_prior[0]) |
                             isinstance(param_prior, list))):
                            for i in range(len(self.wavs)):
                                # Fitted
                                if callable(param_prior[0]):
                                    theta[thetac] = param_prior[i](x[xcount])
                                    xcount += 1
                                # Fixed
                                elif isinstance(param_prior, list):
                                    if isinstance(param_prior[i],
                                                  (int, float)):
                                        theta[thetac] = param_prior[i]
                                    elif isinstance(param_prior[i],
                                                    np.ndarray):
                                        theta[thetac] = np.mean(param_prior[i])
                                else:
                                    raise Exception('Prior syntax error.')
                                thetac += 1

                    # If relationship is set
                    elif (isinstance(param_prior, tuple) &
                          isinstance(param_prior[1], str)):
                        if param_prior[1] == 'polynomial':
                            poly_order = param_prior[2]
                            # Fitted
                            if isinstance(param_prior[0], tuple):
                                polysamps = polynomial(x[xcount:
                                                         xcount+poly_order+1],
                                                       param_prior[0],
                                                       poly_order,
                                                       self.wavs)
                                xcount += poly_order+1
                            # Fixed
                            elif isinstance(param_prior[0], list):
                                polysamps = param_prior[0]
                            else:
                                raise Exception('Prior syntax error.')
                            theta[thetac:thetac+poly_order+1] = polysamps
                            thetac += poly_order+1

                        elif param_prior[1] == 'power-law':
                            powerbounds = param_prior[2]
                            epsilon = param_prior[3]
                            if isinstance(param_prior[0], tuple):
                                plaw_samps = powerlaw(x[xcount:xcount+3],
                                                      param_prior[0],
                                                      self.wavs,
                                                      powerbounds,
                                                      epsilon)
                                xcount += 3
                            elif isinstance(param_prior[0], list):
                                plaw_samps = param_prior[0]
                            else:
                                raise Exception('Prior syntax error.')
                            theta[thetac:thetac+3] = plaw_samps
                            thetac += 3

                        else:
                            raise Exception("Relationship not supported, "
                                            "please choose either "
                                            "'polynomial' or 'power-law'.")
                    else:
                        raise Exception("Invalid prior syntax")

            elif (('psf_pa' in key) |
                  ('bg' in key) |
                  ('rms' in key) |
                  ('counts_per_flux' in key)):

                # Load in the Mimical prior element
                param_prior = self.mimical_prior[key]

                # If same prior for each image
                if ((callable(param_prior) |
                     isinstance(param_prior, (int, float, np.ndarray)))):
                    # Fitted
                    if callable(param_prior):
                        theta[thetac] = param_prior(x[xcount])
                        xcount += 1
                        thetac += 1
                    # Fixed
                    else:
                        if isinstance(param_prior, (int, float)):
                            theta[thetac] = param_prior
                        elif isinstance(param_prior, (np.ndarray)):
                            theta[thetac] = np.mean(param_prior)
                        thetac += 1

                # If different prior for each image
                elif (isinstance(param_prior, (tuple, list)) &
                      (not isinstance(param_prior[1], str))):
                    if ((callable(param_prior[0]) |
                         isinstance(param_prior, list))):
                        for i in range(len(self.wavs)):
                            # Fitted
                            if callable(param_prior[0]):
                                theta[thetac] = param_prior[i](x[xcount])
                                xcount += 1
                            # Fixed
                            elif isinstance(param_prior, list):
                                if isinstance(param_prior[i], (int, float)):
                                    theta[thetac] = param_prior[i]
                                elif isinstance(param_prior[i], np.ndarray):
                                    theta[thetac] = np.mean(param_prior[i])
                            else:
                                raise Exception('Prior syntax error.')
                            thetac += 1

                # If relationship is set
                elif (isinstance(param_prior, tuple) &
                      isinstance(param_prior[1], str)):
                    if param_prior[1] == 'polynomial':
                        poly_order = param_prior[2]
                        # Fitted
                        if isinstance(param_prior[0], tuple):
                            polysamps = polynomial(x[xcount:
                                                     xcount+poly_order+1],
                                                   param_prior[0],
                                                   poly_order,
                                                   self.wavs)
                            xcount += poly_order+1
                        # Fixed
                        elif isinstance(param_prior[0], list):
                            polysamps = param_prior[0]
                        else:
                            raise Exception('Prior syntax error.')
                        theta[thetac:thetac+poly_order+1] = polysamps
                        thetac += poly_order+1

                    elif param_prior[1] == 'power-law':
                        powerbounds = param_prior[2]
                        epsilon = param_prior[3]
                        if isinstance(param_prior[0], tuple):
                            plaw_samps = powerlaw(x[xcount:xcount+3],
                                                  param_prior[0],
                                                  self.wavs,
                                                  powerbounds,
                                                  epsilon)
                            xcount += 3
                        elif isinstance(param_prior[0], list):
                            plaw_samps = param_prior[0]
                        else:
                            raise Exception('Prior syntax error.')
                        theta[thetac:thetac+3] = plaw_samps
                        thetac += 3

                    else:
                        raise Exception("Relationship not supported, "
                                        "please choose either "
                                        "'polynomial' or 'power-law'.")

                else:
                    raise Exception("Invalid prior syntax")

        return theta

    def revert(self, param_dict):
        """ Translate a sampler sample into a sample of model parameters for
        each filter."""

        # Empty parameter array
        params_final = np.zeros((len(self.wavs), np.sum(self.nsources)+4))
        ind = 0
        count = 0

        # Loop over model parameters
        keys = list(self.mimical_prior.keys())
        for i in range(len(keys)):

            if "source" in keys[i]:

                sourcedic = self.mimical_prior[keys[i]]

                for sourcekey in sourcedic.keys():

                    # Load in the Mimical prior element
                    param_prior = sourcedic[sourcekey]

                    if ((callable(param_prior) |
                         isinstance(param_prior, (int, float, np.ndarray)))):
                        params_final[:, ind] = param_dict[count]
                        ind += 1
                        count += 1

                    elif (isinstance(param_prior, (tuple, list)) &
                          (not isinstance(param_prior[1], str))):
                        params_final[:, ind] = param_dict[count:
                                                          count+len(self.wavs)]
                        ind += 1
                        count += len(self.wavs)

                    # If relationship is set
                    elif (isinstance(param_prior, tuple) &
                          isinstance(param_prior[1], str)):
                        if param_prior[1] == 'polynomial':
                            poly_order = param_prior[2]
                            coeffs = param_dict[count:count+poly_order+1]
                            tiler = np.tile(self.wavs-self.wavs[0],
                                            (poly_order+1, 1)).T
                            polywavs = np.pow(tiler, np.arange(poly_order+1))
                            comps = coeffs * polywavs
                            comps_summed = np.sum(comps, axis=1)
                            params_final[:, ind] = comps_summed
                            ind += 1
                            count += poly_order+1

                        elif param_prior[1] == 'power-law':
                            epsilon = param_prior[3]
                            coeffs = param_dict[count:count+3]
                            tiler = np.tile(((self.wavs-self.wavs[0])
                                             + epsilon) /
                                            ((self.wavs[-1]-self.wavs[0])
                                             + epsilon),
                                            (2, 1)).T
                            polywavs = np.pow(tiler, np.array([0, coeffs[2]]))
                            comps = np.array([coeffs[0],
                                             coeffs[1]-coeffs[0]]) * polywavs
                            comps_summed = np.sum(comps, axis=1)
                            params_final[:, ind] = comps_summed
                            ind += 1
                            count += 3

                        else:
                            raise Exception("Relationship not supported, "
                                            "please choose either "
                                            "'polynomial' or 'power-law'.")

            elif (('psf_pa' in keys[i]) |
                  ('bg' in keys[i]) |
                  ('rms' in keys[i]) |
                  ('counts_per_flux' in keys[i])):

                # Load in the Mimical prior element
                param_prior = self.mimical_prior[keys[i]]

                if ((callable(param_prior) |
                     isinstance(param_prior, (int, float, np.ndarray)))):
                    params_final[:, ind] = param_dict[count]
                    ind += 1
                    count += 1

                elif (isinstance(param_prior, (tuple, list)) &
                      (not isinstance(param_prior[1], str))):
                    params_final[:, ind] = param_dict[count:
                                                      count+len(self.wavs)]
                    ind += 1
                    count += len(self.wavs)

                # If relationship is set
                elif (isinstance(param_prior, tuple) &
                      isinstance(param_prior[1], str)):
                    if param_prior[1] == 'polynomial':
                        poly_order = param_prior[2]
                        coeffs = param_dict[count:count+poly_order+1]
                        tiler = np.tile(self.wavs-self.wavs[0],
                                        (poly_order+1, 1)).T
                        polywavs = np.pow(tiler, np.arange(poly_order+1))
                        comps = coeffs * polywavs
                        comps_summed = np.sum(comps, axis=1)
                        params_final[:, ind] = comps_summed
                        ind += 1
                        count += poly_order+1

                    elif param_prior[1] == 'power-law':
                        epsilon = param_prior[3]
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
                        raise Exception("Relationship not supported, "
                                        "please choose either "
                                        "'polynomial' or 'power-law'.")
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
                    param_prior = sourcedic[sourcekey]

                    # If same prior for each image
                    if ((callable(param_prior) |
                         isinstance(param_prior, (int, float)))):
                        if callable(param_prior):
                            nparam += 1
                            ndim += 1
                            keys.append(f'{key}:{sourcekey}')
                            smask.append(True)

                        else:
                            nparam += 1
                            keys.append(f'{key}:{sourcekey}')
                            smask.append(False)

                    # If different prior for each image
                    elif (isinstance(param_prior, (tuple, list)) &
                          (not isinstance(param_prior[1], str))):
                        if ((callable(param_prior[0]) |
                             isinstance(param_prior, list))):
                            for i in range(len(self.wavs)):
                                keys.append(f'{key}:{sourcekey}_'
                                            f'{self.filter_names[i]}')
                                nparam += 1
                                if callable(param_prior[i]):
                                    smask.append(True)
                                    ndim += 1
                                else:
                                    smask.append(False)

                    # If relationship is set
                    elif (isinstance(param_prior, tuple) &
                          isinstance(param_prior[1], str)):
                        if param_prior[1] == 'polynomial':
                            poly_order = param_prior[2]
                            for i in range(0, poly_order+1):
                                keys.append(f'{key}:{sourcekey}_P{i}')
                                nparam += 1
                                if isinstance(param_prior[0], tuple):
                                    smask.append(True)
                                    ndim += 1
                                else:
                                    smask.append(False)

                        elif param_prior[1] == 'power-law':
                            for i in range(3):
                                keys.append(f'{key}:{sourcekey}_PL{i}')
                                nparam += 1
                                if isinstance(param_prior[0], tuple):
                                    smask.append(True)
                                    ndim += 1
                                else:
                                    smask.append(False)
                        else:
                            raise Exception("Relationship not supported, "
                                            "please choose either "
                                            "'Polynomial' or 'Power-law'.")

                    else:
                        raise Exception("Invalid prior syntax")

            elif (('psf_pa' in key) |
                  ('bg' in key) |
                  ('rms' in key) |
                  ('counts_per_flux' in key)):

                # Load in the Mimical prior element
                param_prior = self.mimical_prior[key]

                # If same prior for each image
                if ((callable(param_prior) |
                        isinstance(param_prior, (int, float, np.ndarray)))):
                    if callable(param_prior):
                        nparam += 1
                        ndim += 1
                        keys.append(f'{key}')
                        smask.append(True)

                    else:
                        nparam += 1
                        keys.append(f'{key}')
                        smask.append(False)

                # If different prior for each image
                elif (isinstance(param_prior, (tuple, list)) &
                      (not isinstance(param_prior[1], str))):
                    if ((callable(param_prior[0]) |
                         isinstance(param_prior, list))):
                        for i in range(len(self.wavs)):
                            keys.append(f'{key}_'
                                        f'{self.filter_names[i]}')
                            nparam += 1
                            if callable(param_prior[i]):
                                smask.append(True)
                                ndim += 1
                            else:
                                smask.append(False)

                # If relationship is set
                elif (isinstance(param_prior, tuple) &
                      isinstance(param_prior[1], str)):
                    if param_prior[1] == 'polynomial':
                        poly_order = param_prior[2]
                        for i in range(0, poly_order+1):
                            keys.append(f'{key}_P{i}')
                            nparam += 1
                            if isinstance(param_prior[0], tuple):
                                smask.append(True)
                                ndim += 1
                            else:
                                smask.append(False)

                    elif param_prior[1] == 'power-law':
                        for i in range(3):
                            keys.append(f'{key}_PL{i}')
                            nparam += 1
                            if isinstance(param_prior[0], tuple):
                                smask.append(True)
                                ndim += 1
                            else:
                                smask.append(False)
                    else:
                        raise Exception("Relationship not supported, "
                                        "please choose either "
                                        "'Polynomial' or 'Power-law'.")

                else:
                    raise Exception("Invalid prior syntax")

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

    def plot_prior(self, n, key):
        """ Plot prior samples. """

        samp_filt, keys = self.check_priors(n=n, type='mimical')
        physical = samp_filt

        fig, ax = plt.subplots()

        for i in range(len(physical)):
            curr_sample = physical[i].reshape(len(self.wavs),
                                              len(self.mimical_keys))
            ofinterest = curr_sample[:, self.mimical_keys.index(key)]
            ax.plot(self.wavs, ofinterest, color='black', alpha=0.5)
        ax.set_ylabel(key)
        ax.set_xlabel('$\\lambda$')

        return fig, ax
