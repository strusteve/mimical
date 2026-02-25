import os
from astropy.io import fits
import numpy as np
from astropy.io import ascii

dir_path = os.getcwd()
install_dir = os.path.dirname(os.path.realpath(__file__))
sextractor_dir = (install_dir + "/sextractor_config").replace("/fitting","")


def create_segmaps(id, wavs, images, filter_names, segmaps_empty,  target_maxdistancepix, runtag=''):
    """ Method for cleaning contaminated images with sextractor, overwrites images and segmentation maps. """

    segmaps = segmaps_empty.copy()

    # Make output directories for sextractor output
    if not os.path.isdir(dir_path + "/mimical/sextractor"):
        os.system('mkdir ' + dir_path + "/mimical/sextractor")
        os.system('mkdir ' + dir_path + "/mimical/sextractor/input_images")
        os.system('mkdir ' + dir_path + "/mimical/sextractor/cats")
        os.system('mkdir ' + dir_path + "/mimical/sextractor/segmaps")

    if not os.path.isdir(dir_path + f"/mimical/sextractor/cats{runtag}"):
        os.system('mkdir ' + dir_path + f"/mimical/sextractor/input_images{runtag}")
        os.system('mkdir ' + dir_path + f"/mimical/sextractor/cats{runtag}")
        os.system('mkdir ' + dir_path + f"/mimical/sextractor/segmaps{runtag}")


    # Save images passed to Mimical for passing to Sextractor
    for i in range(len(wavs)):
        hdul = fits.HDUList()
        hdul.append(fits.ImageHDU(data=images[i]))
        hdul.writeto(f"{dir_path}/mimical/sextractor/input_images{runtag}/{id}_{filter_names[i]}.fits", overwrite=True)

    # Run Sextractor
    for i in range(len(wavs)):
        os.system(f"sex {dir_path}/mimical/sextractor/input_images{runtag}/{id}_{filter_names[i]}.fits" +
                    f" -c {sextractor_dir}/jwst_default_segmap.config" +
                    f" -FILTER_NAME {sextractor_dir}/gauss_2.5_5x5.conv" +
                    f" -PARAMETERS_NAME {sextractor_dir}/default.param" +
                    f" -CATALOG_NAME {dir_path}/mimical/sextractor/cats{runtag}/{id}_{filter_names[i]}.cat" +
                    f" -CHECKIMAGE_NAME {dir_path}/mimical/sextractor/segmaps{runtag}/{id}_{filter_names[i]}.fits")
        
    # Loop over filters, load Sextractor catalogues and segmentation maps, determine any areas of contamination and set them to zero.
    for i in range(len(wavs)):
        image = images[i]
        centre_x, centre_y = image.shape[1]/2, image.shape[0]/2
        cat = ascii.read(f"{dir_path}/mimical/sextractor/cats{runtag}/{id}_{filter_names[i]}.cat").to_pandas()
        cat['sep'] = np.sqrt( (cat['X_IMAGE']-centre_x)**2 +  (cat['Y_IMAGE']-centre_y)**2  )
        cat.index = cat['NUMBER'].values

        # If no objects found, leave segmap as ones.
        if len(cat)==0:
            continue

        else:
            segmap = fits.open(f"{dir_path}/mimical/sextractor/segmaps{runtag}/{id}_{filter_names[i]}.fits")[0].data

            # If only one object found
            if len(cat)==1:
                obj_of_interest = cat.iloc[0]

            # If multiple objects found
            else:
                obj_of_interest = cat.loc[cat['NUMBER'].values[np.argmin(cat['sep'])]]

            # If closest object is not near centre, cut it / others
            if obj_of_interest['sep'] > target_maxdistancepix:
                segmap += 1
                segmap[segmap!=1] = 0
                segmaps[i] = segmap

            # If closest object is near centre, cut all else
            else:
                segmap += 1
                segmap[(segmap!=1) & (segmap!=obj_of_interest['NUMBER']+1)] = 0
                segmap[segmap!=0] = 1
                segmaps[i] = segmap
    
    return segmaps