import numpy as np

from .fit import fit
from ..utils import mpi_split_array


class fitCatalogue(object):
    """ Fit a catalogue of singly- or multiply-imaged objects with a 2D model via Bayesian inference.

    Parameters
    ----------

    runtag : str
        A name for the mimical catalogue run.

    id : str
        An ID for the fitting run. Only really used for output files.

    laod_images : function
        A function with input "id" which returns a 3D array of image data with slices for each filter. Each image
        must be the same shape.

    filt_list : str or list
        A function with input "id" which returns a list of path strings to the filter transmission curve files, relative
        to the current working directory. Must be in ascending order with effective wavelength.

    psfs : array
        A function with input "id" which returns a 3D array of normalised PSF images with slices for each filter. Each PSF image
        must be the same shape.

    load_mimical_prior : dict
        A function with input "id" which returns a user specified prior which sets out the priors for 
        the model parameters and passes information about whether to let these vary for each filter 
        or whether they follow an order-specified polynomial relationship.

    astropy_model : array
        Astropy Fittable2DModel used to model the image data. The subsequent prior must include
        only and all parameters in the astropy_model.parameters variable, as well as a 'psf_pa' parameter.
    """

    def __init__(self, runtag, id_list, load_images, load_filt_list, load_psfs, load_mimical_prior, **kwargs):
        
        self.runtag = runtag
        self.id_list = id_list
        self.load_images = load_images
        self.load_filt_list = load_filt_list
        self.load_psfs = load_psfs
        self.load_mimical_prior = load_mimical_prior
        self.kwargs = kwargs

        
    def run(self, mpi_serial=False, make_plots=False):
        """ Runs the nested sampler to sample models, and processes its output.
         
        Parameters
        ----------

        mpi_serial : False
            Whether or not to split ID list among cores, must run script with command
            'mpirun/mpiexec -n [ncores] python [file].
        """

        if not mpi_serial:
            for id in self.id_list:
                single = fit(id, self.load_images(id), self.load_filt_list(id), self.load_psfs(id), self.load_mimical_prior(id), runtag="/"+self.runtag, **self.kwargs)
                single.run()
                if make_plots:
                    single.plot_model()
    
        else:
            id_core = mpi_split_array(np.array((self.id_list)))
            for id in id_core:
                single = fit(id, self.load_images(id), self.load_filt_list(id), self.load_psfs(id), self.load_mimical_prior(id), runtag="/"+self.runtag, **self.kwargs)
                single.run()
                if make_plots:
                    single.plot_model()
        
    
