import os
from astropy.io import fits
import numpy as np
from astropy.io import ascii
import scipy

dir_path = os.getcwd()
install_dir = os.path.dirname(os.path.realpath(__file__))
sextractor_dir = (install_dir +
                  "/config/sextractor_config").replace("/utils", "")


def create_contmaps(id, wavs, images, filter_names, segmaps_empty,
                    target_maxdistancepix, dilute, dilute_radius, runtag=''):
    """ Method for cleaning contaminated images with sextractor, overwrites
    images and segmentation maps. """

    contmaps = segmaps_empty.copy()

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
    # determine any areas of contamination and set them to zero.
    for i in range(len(wavs)):
        image = images[i]
        centre_x, centre_y = (image.shape[1]-1)/2, (image.shape[0]-1)/2
        cat = ascii.read(f"{dir_path}/mimical_output/sextractor/cats{runtag}/"
                         f"{id}_{filter_names[i]}.cat").to_pandas()
        cat['sep'] = np.sqrt(np.array((cat['X_IMAGE']-centre_x)**2 +
                                      (cat['Y_IMAGE']-centre_y)**2))
        cat.index = cat['NUMBER'].values

        # If no objects found, leave segmap as ones.
        if len(cat) == 0:
            continue

        else:
            segmap = fits.open(f"{dir_path}/mimical_output/sextractor/"
                               f"segmaps{runtag}/{id}_{filter_names[i]}.fits"
                               )[0].data.astype(float)
            # If only one object found
            if len(cat) == 1:
                obj_of_interest = cat.iloc[0]

            # If multiple objects found
            else:
                interindex = np.argmin(np.array(cat['sep'].values))
                obj_of_interest = cat.loc[cat['NUMBER'].values[interindex]]

            # If closest object is not near centre, cut it / others
            if obj_of_interest['sep'] > target_maxdistancepix:
                segmap += 1
                segmap[segmap != 1] = 0
                contmaps[i] = segmap

            # If closest object is near centre, cut all else
            else:
                segmap += 1
                segmap[(segmap != 1) &
                       (segmap != obj_of_interest['NUMBER']+1)] = 0
                segmap[segmap != 0] = 1
                contmaps[i] = segmap

    # Dilute the contamination maps usinga circular filter
    if dilute:
        contmaps_diluted = segmaps_empty.copy()
        for i in range(len(contmaps)):
            coordsx, coordsy = np.meshgrid(np.arange(2*dilute_radius+1) -
                                           dilute_radius,
                                           np.arange(2*dilute_radius+1) -
                                           dilute_radius)
            mask = coordsx**2 + coordsy**2 <= dilute_radius**2
            diluted = scipy.ndimage.minimum_filter(contmaps[i], footprint=mask)
            contmaps_diluted[i] = diluted
        return contmaps_diluted

    else:
        return contmaps
