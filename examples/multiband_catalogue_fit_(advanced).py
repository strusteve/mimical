from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from multiprocessing import freeze_support
import mimical
from astropy.table import Table
from photutils.aperture import CircularAperture
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=AstropyWarning)
if __name__ == '__main__':
    freeze_support()

    ############################
    # Define utility functions #
    ############################
    def get_galaxy_image(ra, dec, image_file, fov=5):
        """ Return galaxy cutout at provided coordinates. """
        image = fits.open(image_file)
        wcs = WCS(image[0].header)
        centre = SkyCoord(ra, dec, unit='deg')
        galim = Cutout2D(image[0].data, centre, wcs=wcs,
                         size=u.Quantity((fov, fov), u.arcsec))
        return galim

    def get_flux(image, radius):
        """ Measure flux within circular aperture. """
        ap = CircularAperture(((image.shape[1]-1)/2, (image.shape[0]-1)/2),
                              radius)
        return ap.do_photometry(image)[0][0]

    ############################
    # Define loading functions #
    ############################
    def load_images(id):
        filters = ['f090w', 'f115w', 'f150w', 'f200w',
                   'f277w', 'f356w', 'f410m', 'f444w']
        galaxy_catalogue = Table.read('galaxy_catalogue',
                                      format="ascii.cds").to_pandas()
        galaxy_catalogue.index = galaxy_catalogue['ID'].values.astype(str)
        images = []
        for i in range(len(filters)):
            image = get_galaxy_image(galaxy_catalogue.loc[id, 'RA'],
                                     galaxy_catalogue.loc[id, 'DEC'],
                                     f'mosaic_{filters[i]}.fits').data
            image[np.isnan(image)] = 0
            images.append(image)
        return images

    def load_psfs(id):
        filters = ['f090w', 'f115w', 'f150w', 'f200w',
                   'f277w', 'f356w', 'f410m', 'f444w']
        psfs = []
        for i in range(len(filters)):
            psfi = np.load(f'psf_{filters[i]}.npy')
            psfs.append(psfi)
        return psfs

    def load_filt_list(id):
        filters = ['f090w', 'f115w', 'f150w', 'f200w',
                   'f277w', 'f356w', 'f410m', 'f444w']
        return ['filters/'+filt.upper() for filt in filters]

    def load_mimical_prior(id):
        filters = ['f090w', 'f115w', 'f150w', 'f200w',
                   'f277w', 'f356w', 'f410m', 'f444w']
        galaxy_catalogue = Table.read('paper_2_sizemass/'
                                      'Stevenson2025_Table3.txt',
                                      format="ascii.cds").to_pandas()
        galaxy_catalogue.index = galaxy_catalogue['ID'].values.astype(str)

        images = load_images(id)
        cpfs = []
        for filt in filters:
            cpfi = get_galaxy_image(galaxy_catalogue.loc[id, 'RA'],
                                    galaxy_catalogue.loc[id, 'DEC'],
                                    f'paper_2_sizemass/exposure_maps.nosync'
                                    f'/expmap_{filt}_only1837.fits').data
            cpfs.append(cpfi * 28e6)

        mimical_prior = {}

        source_1 = {}
        # Vary 'flux' from 0 to the maximum measured flux.
        maxflux = np.max([(10*get_flux(imp, 0.5/0.03)) for imp in images])
        source_1['flux'] = ((0, maxflux), 'Individual')
        # Vary 'r_eff' from 0 to 20. Assume power-law relationship from
        # index -5 to 5.
        source_1['r_eff'] = ((0, 20), 'Power-law', (-5, 5))
        # Vary 'n' from 0.1 to 10.
        source_1['n'] = ((0.1, 10), 'Polynomial', 1)
        # Vary 'x_0' in centred box of length 40.
        source_1['x_0'] = ((images[0].shape[1]/2-20, images[0].shape[1]/2+20),
                           'Polynomial', 0)
        # Vary 'y_0' in centred box of length 40.
        source_1['y_0'] = ((images[0].shape[0]/2-20, images[0].shape[0]/2+20),
                           'Polynomial', 0)
        # Vary 'ellip' from 0 to 1.
        source_1['ellip'] = ((0, 1), 'Polynomial', 0)
        # Vary 'theta' from 0 to Pi.
        source_1['theta'] = ((0, np.pi), 'Polynomial', 0)

        mimical_prior['source_1'] = source_1
        # Fix 'psf_pa' to 0, no rotation.
        mimical_prior['psf_pa'] = (0, 'Individual')
        # Infer 'rms' using SourceExtractor.
        mimical_prior['rms'] = ('Infer', 'Individual')
        # Fix 'counts_per_flux' to provided values.
        mimical_prior['counts_per_flux'] = (cpfs, 'Individual')

        return mimical_prior

    #####################
    # Run catalgoue fit #
    #####################
    cat = Table.read('galaxy_catalogue', format="ascii.cds").to_pandas()
    cat.index = cat['ID'].values.astype(str)
    cat['ID'] = cat.index.values
    id_list = cat.index.values[cat['Survey'] == 'PRIMER-UDS']

    fit = mimical.fitCatalogue('examplecat', id_list, load_images,
                               load_filt_list, load_psfs, load_mimical_prior,
                               se_clean=True)
    # Automatic oversampling. Perform parallelisation of individual fits to
    # individual cores. Script must be run with
    # 'mpirun -n XXX python this_script.py'
    fit.run(oversample='auto', mpi_serial=True, verbose_sampler=False)
    fit.plot_model()
