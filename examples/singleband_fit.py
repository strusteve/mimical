from multiprocessing import freeze_support
from astropy.io import fits
import numpy as np
import mimical
if __name__ == '__main__':
    freeze_support()

    #############
    # Load data #
    #############
    # The filter to fit, not hugely important in a single fit.
    filters = ['f090w']
    # This assumes you have filter transmission files in folder 'filters/'
    filt_list = ['filters/'+filt.upper() for filt in filters]
    # Load NxM image
    image = [fits.open(f'image_f090w.fits').data]
    # Load PSF
    psf = [fits.open(f'psf_f090w.fits')]
    # Load counts-per-flux image. Here 28e6 is the counts-per-second-per-flux.
    # Mutliplied by the exposure time, this gives counts-per-flux.
    cpf = [fits.open(f'exposuremap_f090w.fits').data * 28e6]

    ########################
    # Define mimical prior #
    ########################
    mimical_prior = {}

    # Define a Sersic source
    source_1 = {}
    # Vary 'flux' from 0 to 1. Must chose 'Individual' option for
    # single-band fits.
    source_1['flux'] = ((0, 1), 'Individual')
    # Vary 'r_eff' from 1 to 20. Must chose 'Individual' option for
    # single-band fits.
    source_1['r_eff'] = ((0, 20), 'Individual')
    # Vary 'n' from 0.1 to 10. Must chose 'Individual' option for
    # single-band fits.
    source_1['n'] = ((0.1, 10), 'Individual')
    # Vary 'x_0' in a box of length 40 pixels centred on the image. Must chose
    # 'Individual' option for single-band fits.
    source_1['x_0'] = ((image[0].shape[1]/2-20, image[0].shape[1]/2+20),
                       'Individual')
    # Vary 'y_0' in a box of length 40 pixels centred on the image. Must chose
    # 'Individual' option for single-band fits.
    source_1['y_0'] = ((image[0].shape[0]/2-20, image[0].shape[0]/2+20),
                       'Individual')
    # Vary 'ellip' from 0 to 1. Must chose 'Individual' option for
    # single-band fits.
    source_1['ellip'] = ((0, 1), 'Individual')
    # Vary 'theta' from 0 to Pi. Must chose 'Individual' option for
    # single-band fits.
    source_1['theta'] = ((0, np.pi), 'Individual')

    # Add Sersic source to mimical prior.
    mimical_prior['source_1'] = source_1
    # Fix 'psf_pa' to 0. This assumes an empirical PSF, and therefore no
    # rotation is performed, improving computation time.
    mimical_prior['psf_pa'] = (0, 'Individual')
    # Infer the 'rms' from its SourceExtracted background. This is only
    # calculated once and hance speeds up computation time.
    mimical_prior['rms'] = ('Infer', 'Individual')
    # Fix the 'counts-per-flux' to the provided image. This allows the
    # inclusion of poisson noise, yielding more realistic posteriors.
    mimical_prior['counts_per_flux'] = (cpf, 'Individual')

    ###############
    # Run mimical #
    ###############
    # Use SourceExtractor to find background (required for 'Infer' of 'rms')
    # and dilute its segmentation map.
    fit = mimical.fit('example', image, filt_list, psf, mimical_prior,
                      se_clean=True, dilute=True)
    # Oversample the model in annulii, perform no parallelisation of
    # acceleration.
    fit.run(oversample=[20, 10, 5], oversample_radii=[1, 10, 20],
            pool=None, gpu_acceleration=False)
    # Save plots of the results.
    fit.plot_model()
