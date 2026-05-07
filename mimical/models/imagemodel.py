from scipy.ndimage import rotate
from scipy.signal import convolve
import numpy as np
from astropy.nddata import block_reduce

class ImageModel(object):
    """ Base class for evaluating a parametric sub-model on a pixel grid and convolving with a PSF """

    def __init__(self, model, psf, oversampling_radii=0, oversampling_factors=1):

        self.model = model
        self.psf = psf
        self.oversampling_radii = oversampling_radii
        self.oversampling_factors = oversampling_factors

        self.param_names = [*self.model.param_names, 'psf_pa']


    def update_parameters(self, parameters):

        *self.model_params, self.psf_pa = parameters
        self.model.update_parameters(self.model_params)


    def render(self, x, y):
        
        # Evaluate submodel over grid
        model_image = self.evaluate_over_grid(x, y, self.oversampling_radii, self.oversampling_factors)

        # Convolve submodel image with PSF image
        final_image = self.PSFconvolve(model_image)

        return final_image
    

    def evaluate_over_grid(self, x, y, oversampling_radii=0, oversampling_factors=1):

        # Make pixel grid
        base_xgrid, base_ygrid = np.meshgrid(x, y)

        # If no oversampling specified
        if oversampling_factors == 1:
            return self.model.evaluate(base_xgrid, base_ygrid)

        # If oversampling specified, prepare base pixel coord grid
        model_image = np.zeros((len(y), len(x)))
        centred_base_xgrid = base_xgrid - ((len(x)-1) / 2)
        centred_base_ygrid = base_ygrid - ((len(y)-1) / 2)

        # For homogeneous oversampling, expand the pixel grid and then block-reduce
        if isinstance(oversampling_factors, (int, float)):
            x_arange = (np.arange(0.5, len(x) * oversampling_factors, 1) / oversampling_factors) - 0.5
            y_arange = (np.arange(0.5, len(y) * oversampling_factors, 1) / oversampling_factors) - 0.5
            xgrid, ygrid = np.meshgrid(x_arange, y_arange)
            evaluation = self.model.evaluate(xgrid, ygrid)
            model_image = block_reduce(evaluation, oversampling_factors) / oversampling_factors**2

        # For inhomogeneous oversampling, loop over annuli about the image centre
        else:

            # Loop over oversampling radii
            for i in range(0, len(oversampling_radii)):
                
                # If first radii, include centre
                if i == 0:
                    curr_mask = (centred_base_xgrid**2 + centred_base_ygrid**2 <= oversampling_radii[i]**2)
                # Else, mask in annuli
                else:
                    curr_mask = (centred_base_xgrid**2 + centred_base_ygrid**2 <= oversampling_radii[i]**2) & (centred_base_xgrid**2 + centred_base_ygrid**2 > oversampling_radii[i-1]**2)

                # Evaluate the oversampling pixel coord 'shift', aka for an oversample factor of 4, this will be [-0.375 -0.125  0.125  0.375]
                oversample_shift = (np.arange(oversampling_factors[i]) - ((oversampling_factors[i]-1)/2)) * (1/oversampling_factors[i])

                # Make oversampled sub-pixel coord grid
                oversampled_xgrid_tiles = np.tile(base_xgrid[curr_mask],(oversampling_factors[i],1)).T
                oversampled_xgrid_coords = (oversampled_xgrid_tiles + oversample_shift)
                oversampled_ygrid_tiles = np.tile(base_ygrid[curr_mask],(oversampling_factors[i],1)).T
                oversampled_ygrid_coords = (oversampled_ygrid_tiles + oversample_shift)

                # Manually meshgrid subpixel coords
                oversampled_xgrid = np.tile(oversampled_xgrid_coords, (1, oversampling_factors[i]))
                oversampled_ygrid = np.repeat(oversampled_ygrid_coords, oversampling_factors[i], axis=1)

                '''
                # Plot the sub-pixel coord grid
                plt.scatter(oversampled_xgrid, oversampled_ygrid, marker='+')
                for i in range(len(x)):
                    plt.axvline(i+0.5,0,1, color='black', lw=1)
                    plt.axhline(i+0.5,0,1, color='black', lw=1)
                plt.show()
                '''

                # Evaluate over sub-pixel grid
                evaluation = self.model.evaluate(oversampled_xgrid, oversampled_ygrid)

                # Downsample the evaluated grid to the pixel scale
                downsample_evaluation = np.sum(evaluation, axis=1) / oversampling_factors[i]**2

                # Append values to base image grid
                model_image[curr_mask] = downsample_evaluation

            # Evaluate any pixels outside specified radii, if any
            remaining_mask = centred_base_xgrid**2 + centred_base_ygrid**2 > oversampling_radii[-1]**2
            if remaining_mask.any():
                remaining_xgrid = base_xgrid[remaining_mask]
                remaining_ygrid = base_ygrid[remaining_mask]
                evaluate_remaining = self.model.evaluate(remaining_xgrid, remaining_ygrid)
                model_image[remaining_mask] = evaluate_remaining
        
        return model_image


    def PSFconvolve(self, model_image):
        
        # Rotate PSF if rotation
        if self.psf_pa != 0:
            psf = rotate(self.psf, self.psf_pa, reshape=False)
        else:
            psf = self.psf
        
        # Convolve PSF
        convolved_image = convolve(model_image, psf, mode="same")

        return convolved_image
            
        

