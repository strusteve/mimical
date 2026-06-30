from multiprocessing import freeze_support
from astropy.io import fits
import numpy as np
import mimical
if __name__ == '__main__':
    freeze_support()

    #############
    # Load data #
    #############
    # All filters fit
    filters = ['f090w', 'f115w', 'f150w', 'f200w',
               'f277w', 'f356w', 'f410m', 'f444w']
    # This assumesyou have filter transmission files in folder 'filters/'
    filt_list = ['filters/'+filt.upper() for filt in filters]

    images = []
    psfs = []
    cpfs = []
    for filt in filters:
        # Load NxM image
        image = fits.open(f'image_{filt}.fits').data
        # Load PSF
        psfi = fits.open(f'psf_{filt}.fits')
        # Load counts-per-flux image. Here 28e6 is the
        # counts-per-second-per-flux. Mutliplied by the exposure time,
        # this gives counts-per-flux.
        cpfi = fits.open(f'exposuremap_{filt}.fits').data * 28e6

        images.append(image)
        psfs.append(psfi)
        cpfs.append(cpfi)

    ########################
    # Define mimical prior #
    ########################
    mimical_prior = {}

    # Define a Sersic source
    source_1 = {}
    # Vary 'flux' in each filter from 0 to 1. Assume no relationship, free
    # parameter for each filter.
    source_1['flux'] = ((0, 1), 'Individual')
    # Vary 'r_eff' in each filter from 1 to 20. Assume a power-law relationship
    # from index -3 to 3, three free parameters.
    source_1['r_eff'] = ((0, 20), 'Power-law', (-5, 5))
    # Vary 'n' from 0.1 to 10 in each filter. Assume a polynomial relationship
    # of rank 1 (straight-line), 2 free parameters.
    source_1['n'] = ((0.1, 10), 'Polynomial', 1)
    # Vary 'x_0' in a box of length 40 pixels centred on the image for each
    # filter. Assume a polynomial relationship of rank 0 (constant).
    # 1 free parameter.
    source_1['x_0'] = ((image[0].shape[1]/2-20, image[0].shape[1]/2+20),
                       'Polynomial', 0)
    # Vary 'y_0' in a box of length 40 pixels centred on the image for each
    # filter. Assume a polynomial relationship of rank 0 (constant).
    # 1 free parameter.
    source_1['y_0'] = ((image[0].shape[0]/2-20, image[0].shape[0]/2+20),
                       'Polynomial', 0)
    # Vary 'ellip' from 0 to 1 for each filter. Assume a polynomial
    # relationship of rank 0 (constant). 1 free parameter.
    source_1['ellip'] = ((0, 1), 'Polynomial', 0)
    # Vary 'theta' from 0 to Pi for each filter. Assume a polynomial
    # relationship of rank 0 (constant). 1 free parameter.
    source_1['theta'] = ((0, np.pi), 'Polynomial', 0)

    # Add Sersic source to mimical prior
    mimical_prior['source_1'] = source_1
    # Fix 'psf_pa' to 0 for each filters. This assumes an empirical PSF,
    # and therefore no rotation is performed, improving computation time.
    mimical_prior['psf_pa'] = (0, 'Individual')
    # Infer the 'rms' for each filter from its SourceExtracted background.
    # This is only calculated once and hance speeds up computation time.
    mimical_prior['rms'] = ('Infer', 'Individual')
    # Fix the 'counts-per-flux' to the provided images. This allows the
    # inclusion of poisson noise, yielding more realistic posteriors.
    mimical_prior['counts_per_flux'] = (cpfs, 'Individual')

    ###############
    # Run mimical #
    ###############
    # Use SourceExtractor to find background (required for 'Infer' of 'rms')
    # and dilute its segmentation map.
    fit = mimical.fit('example', images, filt_list, psfs, mimical_prior,
                      se_clean=True, dilute=True)
    # Oversample the model in annulii, perform no parallelisation of
    # acceleration.
    fit.run(oversample=[20, 10, 5], oversample_radii=[1, 10, 20],
            pool=None, gpu_acceleration=False)
    # Save plots of the results.
    fit.plot_model()
