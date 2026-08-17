import numpy as np
import os

from .fit import fit
from ..utils import mpi_split_array


class fitCatalogue(object):
    """ Fit a catalogue of singly- or multiply-imaged objects with a 2D model
        via Bayesian inference.

    Parameters
    ----------

    runtag : str
        A name for the mimical catalogue run.

    id_list : list
        An ID list for the fitting run. Only really used for output files.

    load_images : function
        Function taking in 'id' and returning a list of images with slices for
        each filter.

    load_filt_list : function
        Function taking in 'id' and returning a  list of path strings to the
        filter transmission curve files, relative to the current working
        directory.

    load_psfs : function
        Function taking in 'id' and returning a list of PSF images with slices
        for each filter.

    load_mimical_prior : function
        The user specified prior which set out the priors for the model
        parameters and passes information about whether to let these vary for
        each filter or whether they follow a power-law or an order-specified
        polynomial relationship.
    """

    def __init__(self, runtag, id_list, load_images, load_filt_list, load_psfs,
                 load_mimical_prior, **kwargs):

        self.runtag = runtag
        self.id_list = id_list
        self.load_images = load_images
        self.load_filt_list = load_filt_list
        self.load_psfs = load_psfs
        self.load_mimical_prior = load_mimical_prior
        self.kwargs = kwargs

    def run(self, mpi_serial=False, make_plots=False, **run_kwargs):
        """ Runs the nested sampler to sample models, and processes its output.

        Parameters
        ----------

        mpi_serial : bool
            Whether or not to split ID list among cores, must run script with
            command 'mpirun/mpiexec -n [ncores] python [file]'

        n_live : int
            Number of live points in nested sampling algorithm.

        make_plots : bool
            Save key plots.
        """

        if not mpi_serial:
            for id in self.id_list:
                single = fit(id, self.load_images(id), self.load_filt_list(id),
                             self.load_psfs(id), self.load_mimical_prior(id),
                             runtag="/"+self.runtag, **self.kwargs)
                single.run(**run_kwargs)
                if make_plots:
                    single.save_plots()

        else:
            id_core, rank = mpi_split_array(np.array((self.id_list)))
            for id in id_core:
                single = fit(id, self.load_images(id), self.load_filt_list(id),
                             self.load_psfs(id), self.load_mimical_prior(id),
                             runtag="/"+self.runtag, rank=f'_core{rank}', **self.kwargs)
                single.run(**run_kwargs)
                if make_plots:
                    single.save_plots()
