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

from .fit import fit
from .prior_handler import priorHandler
from ..plotting import plotter
from ..utils import filter_set
from ..utils import mpi_split_array


class fitCatalogue(object):
    """ Mimical is an intensity modelling code for multiply-imaged objects, 
    performing simultaenous Bayseian inference of model parameters via the 
    nested sampling algorithm. Mimical supports any astropy 2D model, and 
    supports user defined parameter polynomial depenency with image wavelength.

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

        
    def run(self, mpi_serial=False, make_plots=False, plot_type='median'):
        """ Runs the nested sampler to sample models, and processes its output.
         
        Parameters
        ----------

        mpir_serial : False
            Whether or not to split ID list among cores, must run script with command
            'mpirun/mpiexec -n [ncores] python [file].
        """

        if not mpi_serial:
            for id in self.id_list:
                single = fit(id, self.load_images(id), self.load_filt_list(id), self.load_psfs(id), self.load_mimical_prior(id), **self.kwargs)
                single.run(runtag="/"+self.runtag)
                if make_plots:
                    single.plot_model(type=plot_type, runtag="/"+self.runtag)
    
        else:
            id_core = mpi_split_array(np.array((self.id_list)))
            for id in id_core:
                single = fit(id, self.load_images(id), self.load_filt_list(id), self.load_psfs(id), self.load_mimical_prior(id), **self.kwargs)
                single.run(runtag="/"+self.runtag)
        
        
    
