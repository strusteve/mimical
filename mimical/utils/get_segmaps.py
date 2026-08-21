import os
from astropy.io import fits
import numpy as np
from astropy.io import ascii
import scipy

dir_path = os.getcwd()
install_dir = os.path.dirname(os.path.realpath(__file__))
sextractor_dir = (install_dir +
                  "/config/sextractor_config").replace("/utils", "")


def get_segmaps(id, wavs, images, filter_names, se_maxdist, runtag=''):
    """ Method for cleaning contaminated images with sextractor, overwrites
    images and segmentation maps. """

    segmaps = []

    if not os.path.isdir(dir_path +
                         f"/mimical_output/sextractor/cats{runtag}"):
        os.system('mkdir -p ' + dir_path +
                  f"/mimical_output/sextractor/input_images{runtag}")
        os.system('mkdir -p ' + dir_path +
                  f"/mimical_output/sextractor/cats{runtag}")
        os.system('mkdir -p ' + dir_path +
                  f"/mimical_output/sextractor/segmaps{runtag}")

    # Save images passed to Mimical for passing to Sextractor
    for i in range(len(wavs)):
        hdul = fits.HDUList()
        hdul.append(fits.ImageHDU(data=images[i]))
        hdul.writeto(f"{dir_path}/mimical_output/sextractor/input_images"
                     f"{runtag}/{id}_{filter_names[i]}.fits", overwrite=True)

    # Run Sextractor
    for i in range(len(wavs)):
        os.system(f"sex {dir_path}/mimical_output/sextractor/"
                  f"input_images{runtag}/{id}_{filter_names[i]}.fits -c "
                  f"{sextractor_dir}/jwst_default_segmap.config -FILTER_NAME "
                  f"{sextractor_dir}/gauss_2.5_5x5.conv -PARAMETERS_NAME "
                  f"{sextractor_dir}/default.param -CATALOG_NAME {dir_path}/"
                  f"mimical_output/sextractor/cats{runtag}/{id}_"
                  f"{filter_names[i]}.cat -CHECKIMAGE_NAME {dir_path}/"
                  f"mimical_output/sextractor/segmaps{runtag}/{id}_"
                  f"{filter_names[i]}.fits "
                  "> /dev/null 2>&1")

    # Loop over filters, load Sextractor catalogues and segmentation maps,
    # determine any areas of contamination.
    for i in range(len(wavs)):
        image = images[i]
        centre_x, centre_y = (image.shape[1]-1)/2, (image.shape[0]-1)/2
        cat = ascii.read(f"{dir_path}/mimical_output/sextractor/cats{runtag}/"
                         f"{id}_{filter_names[i]}.cat").to_pandas()
        cat['sep'] = np.sqrt(np.array((cat['X_IMAGE']-centre_x)**2 +
                                      (cat['Y_IMAGE']-centre_y)**2))
        cat.index = cat['NUMBER'].values

        segmap = fits.open(f"{dir_path}/mimical_output/sextractor/"
                           f"segmaps{runtag}/{id}_{filter_names[i]}.fits"
                           )[0].data.astype(float)

        # Set object of interest
        if (len(cat) == 0):
            obj_of_interest = None
        elif len(cat) == 1:
            obj_of_interest = cat.iloc[0]
        else:
            interindex = np.argmin(np.array(cat['sep'].values))
            obj_of_interest = cat.loc[cat['NUMBER'].values[interindex]]

        # If closest object is not near centre, cut it / others
        if obj_of_interest['sep'] > se_maxdist:
            segmap[segmap != 0] = 2

        # If closest object is near centre, cut all else
        else:
            segmap[segmap == obj_of_interest['NUMBER']] = 1
            segmap[(segmap != 0) & (segmap != obj_of_interest['NUMBER'])] = 2

        segmaps.append(segmap)

    return segmaps


def dilute_segmaps(segmaps, dilute_radius):

    segmaps_diluted = []
    for i in range(len(segmaps)):
        coordsx, coordsy = np.meshgrid(np.arange(2*dilute_radius+1) -
                                       dilute_radius,
                                       np.arange(2*dilute_radius+1) -
                                       dilute_radius)
        mask = coordsx**2 + coordsy**2 <= dilute_radius**2
        diluted = scipy.ndimage.minimum_filter(segmaps[i], footprint=mask)
        segmaps_diluted.append(diluted)

    return segmaps_diluted
