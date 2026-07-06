import torch
import warnings
import matplotlib.pyplot as plt
# Filter out the specific dynamic resize warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="An output with one or more elements was resized")

from .rotationmodels import Rotator

class ImageModel(object):
    """ Base class for evaluating a parametric sub-model on a pixel grid.

    Parameters
    ----------

    x : 1dtensor
        The x-axis coordinates of the image
        (e.g. torch.arange(image.shape[1])).

    y : 1dtensor
        The y-axis coordinates of the image
        (e.g. torch.arange(image.shape[0])).

    submodels : list
        List of submodels used in the fit (e.g. for two component Sersic this
        is [Sersic(), Sersic()]).

    psf : 1dtensor or 2dtensor
        A 2D PSF or 3D PSF with slices for each filter.

    psf_pa : 1dtensor
        A 1D tensor of PSF position angles (east of north) with elements for
        each filter.

    oversample : int or list of ints
        The factor by which to oversample the image. Integer for 'homogeneous'
        and 'window' methods, list for 'annuli' method.

    oversample_boxlength : int
        Length of centred square in which to oversample.

    oversample_radii : list of floats
        Radii of annuli in which to oversample.
    """

    def __init__(self, x, y, submodels, psf, psf_pa, oversample=None,
                 oversample_boxlength=None, oversample_radii=None):

        self.x = x
        self.y = y
        self.nsources = [i.params.shape[1] for i in submodels]
        self.submodels = submodels
        self.psf = psf
        self.oversample = oversample
        self.oversample_boxlength = oversample_boxlength
        self.oversample_radii = oversample_radii
        self.psf_pa = psf_pa

        # Cache image coord grid for efficiency
        self.base_xgrid, self.base_ygrid = torch.meshgrid(x, y, indexing='xy')

        # Cache PSF coord grid for efficiency
        if self.psf is not None:
            psfx = torch.arange(psf[0].shape[1], device=self.x.device)
            psfy = torch.arange(psf[0].shape[0], device=self.x.device)
            self.psf_xgrid, self.psf_ygrid = torch.meshgrid(psfx, psfy,
                                                            indexing='xy')
            self.utilnum = torch.arange(len(psf_pa), device=x.device)

        self.rot = Rotator(device=self.x.device)

    def update_parameters(self, submodel_params, psf_pa):
        """ Update the submodel parameters and psf position angles. """

        for i in range(len(self.nsources)):
            if i == 0:
                pars = submodel_params[:, :self.nsources[i]]
                self.submodels[i].update_parameters(pars)
            else:
                partrack = torch.sum(torch.tensor(self.nsources[:i]))
                pars = submodel_params[:,
                                       partrack:
                                       partrack+self.nsources[i]]
                self.submodels[i].update_parameters(pars)
        self.psf_pa = psf_pa

    def update_oversampling(self, oversample=None, oversample_boxlength=None,
                            oversample_radii=None):
        """ Update the oversampling parameters. """

        self.oversample = oversample
        self.oversample_boxlength = oversample_boxlength
        self.oversample_radii = oversample_radii

    def render(self):
        """ Render the ImageModel onto its pixel grid. """

        # Evaluate submodel over the pixel grid
        model_image = self.evaluate_over_grid(self.submodels[0],
                                              self.oversample,
                                              self.oversample_boxlength,
                                              self.oversample_radii)

        # For multiple sources in scene
        if len(self.nsources) > 1:
            for i in range(1, len(self.nsources)):
                newsource = self.evaluate_over_grid(self.submodels[i],
                                                    self.oversample,
                                                    self.oversample_boxlength,
                                                    self.oversample_radii)
                model_image += newsource

        # If no PSF is provided, return base model
        if self.psf is None:
            return model_image

        # If PSF is provided, rotate if desired then convolve
        else:
            # Rotation
            if (self.psf_pa == 0).all():
                psf_rot = self.psf
            else:
                psf_rot = self.rot.intersection(self.psf, self.psf_pa,
                                                base_x=self.psf_xgrid,
                                                base_y=self.psf_ygrid,
                                                utilnum=self.utilnum)
                psf_rot = (psf_rot.T / torch.sum(psf_rot, (1, 2))).T

            # Convolve submodel image with PSF image
            final_image = self.PSFconvolve(model_image, psf_rot)
            return final_image

    def evaluate_over_grid(self, model, oversample, oversample_boxlength,
                           oversample_radii):
        """ Evaluate submodel over the pixel grid. """

        # If no oversampling specified
        if oversample is None:
            # Make pixel grid
            full_xgrid = torch.stack(([self.base_xgrid]*model.params.shape[0]))
            full_ygrid = torch.stack(([self.base_ygrid]*model.params.shape[0]))
            return model.evaluate(full_xgrid, full_ygrid)

        # Expand the pixel grid and then block-reduce
        elif (isinstance(oversample, (int, float)) &
              (oversample_boxlength is None) &
              (oversample_radii is None)):
            return self.homogeneous_oversampling(model, oversample)

        # Expand the window pixel grid and then block-reduce
        elif (isinstance(oversample, (int, float)) &
              isinstance(oversample_boxlength, (int, float))
              & (oversample_radii is None)):
            return self.window_oversampling(model, oversample,
                                            oversample_boxlength)

        # For inhomogeneous oversampling, loop over annuli about image centre
        elif (isinstance(oversample, list) &
              isinstance(oversample_radii, list) &
              (oversample_boxlength is None)):
            return self.annuli_oversampling(model, oversample,
                                            oversample_radii)

        else:
            raise Exception("Invalid syntax")

    def homogeneous_oversampling(self, model, oversample):
        """ Oversample the entire image by the factor provided. """

        oversample_shift = (torch.arange(oversample, device=self.x.device) -
                            ((oversample-1) / 2)) * (1 / oversample)

        # Make oversampled sub-pixel coord grid
        oversampled_x_tiles = torch.tile(self.base_xgrid.flatten(),
                                         (oversample, 1)).T
        oversampled_x_coo = (oversampled_x_tiles + oversample_shift)
        oversampled_y_tiles = torch.tile(self.base_ygrid.flatten(),
                                         (oversample, 1)).T
        oversampled_y_coo = (oversampled_y_tiles + oversample_shift)

        # Manually meshgrid subpixel coords
        oversampled_xgrid = torch.tile(oversampled_x_coo, (1, oversample))
        oversampled_ygrid = torch.repeat_interleave(oversampled_y_coo,
                                                    oversample, axis=1)

        full_xgrid = torch.stack(([oversampled_xgrid]*model.params.shape[0]))
        full_ygrid = torch.stack(([oversampled_ygrid]*model.params.shape[0]))

        evaluation = model.evaluate(full_xgrid, full_ygrid)

        downsampled_evaluation = torch.sum(evaluation, dim=2)

        return downsampled_evaluation.reshape(model.params.shape[0],
                                              len(self.y),
                                              len(self.x)) / oversample**2

    def window_oversampling(self, model, oversample, oversample_boxlength):
        """ Oversampled a centred window of length 'oversample_boxlength'. """

        model_image = torch.zeros(model.params.shape[0],
                                  *self.base_xgrid.shape,
                                  device=self.x.device)

        # Inside box
        boxinmask = self.base_xgrid != self.base_xgrid
        boxinmask[(self.y.shape[0]-oversample_boxlength)//2:
                  -(self.y.shape[0]-oversample_boxlength)//2,
                  (self.x.shape[0]-oversample_boxlength)//2:
                  -(self.x.shape[0]-oversample_boxlength)//2] = True
        xgrid_inbox = self.base_xgrid[boxinmask]
        ygrid_inbox = self.base_ygrid[boxinmask]

        oversample_shift = (torch.arange(oversample, device=self.x.device) -
                            ((oversample-1) / 2)) * (1 / oversample)

        # Make oversampled sub-pixel coord grid
        oversampled_xgrid_tiles = torch.tile(xgrid_inbox, (oversample, 1)).T
        oversampled_xgrid_coords = (oversampled_xgrid_tiles + oversample_shift)
        oversampled_ygrid_tiles = torch.tile(ygrid_inbox, (oversample, 1)).T
        oversampled_ygrid_coords = (oversampled_ygrid_tiles + oversample_shift)

        # Manually meshgrid subpixel coords
        oversampled_xgrid = torch.tile(oversampled_xgrid_coords,
                                       (1, oversample))
        oversampled_ygrid = torch.repeat_interleave(oversampled_ygrid_coords,
                                                    oversample, axis=1)
        full_xgrid = torch.stack(([oversampled_xgrid]*model.params.shape[0]))
        full_ygrid = torch.stack(([oversampled_ygrid]*model.params.shape[0]))

        # Evaluate and downsample
        evaluation = model.evaluate(full_xgrid, full_ygrid)
        inbox_evaluation = torch.sum(evaluation, dim=2) / oversample**2

        # Outside box
        xgrid_outbox = self.base_xgrid[~boxinmask]
        ygrid_outbox = self.base_ygrid[~boxinmask]
        full_xgrid = torch.stack(([xgrid_outbox] *
                                  model.params.shape[0])).unsqueeze(-1)
        full_ygrid = torch.stack(([ygrid_outbox] *
                                  model.params.shape[0])).unsqueeze(-1)
        outbox_evaluation = model.evaluate(full_xgrid, full_ygrid)

        # Combine all
        boxinmask = torch.broadcast_to(boxinmask,
                                       (model.params.shape[0],
                                        *boxinmask.shape))
        model_image[boxinmask] = inbox_evaluation.flatten()
        model_image[~boxinmask] = outbox_evaluation.flatten()

        return model_image

    def annuli_oversampling(self, model, oversample, oversample_radii):
        """ Oversampling in annulii by the factors 'oversample'. """

        # Make a centred coordinate grid
        model_image = torch.zeros(model.params.shape[0],
                                  *self.base_xgrid.shape,
                                  device=self.x.device)
        # centred_base_xgrid = self.base_xgrid - ((self.x.shape[0]-1) / 2)
        # centred_base_ygrid = self.base_ygrid - ((self.y.shape[0]-1) / 2)
        centred_base_xgrid = self.base_xgrid - \
            torch.mean(self.submodels[0].x_0)
        centred_base_ygrid = self.base_ygrid - \
            torch.mean(self.submodels[0].y_0)

        # Loop over oversampling radii
        for i in range(0, len(oversample_radii)):

            # If first radii, include centre
            if i == 0:
                curr_mask = (centred_base_xgrid**2 + centred_base_ygrid**2 <=
                             oversample_radii[i]**2)

            # Else, mask in annuli
            else:
                curr_mask = (centred_base_xgrid**2 + centred_base_ygrid**2 <=
                             oversample_radii[i]**2) & \
                            (centred_base_xgrid**2 + centred_base_ygrid**2 >
                             oversample_radii[i-1]**2)

            # If oversample is 1, skip.
            if oversample[i] == 1:
                # Evaluate over sub-pixel grid
                raw_xgrid = self.base_xgrid[curr_mask].unsqueeze(-1)
                raw_ygrid = self.base_ygrid[curr_mask].unsqueeze(-1)
                full_raw_xgrid = torch.stack(([raw_xgrid] *
                                              model.params.shape[0]))
                full_raw_ygrid = torch.stack(([raw_ygrid] *
                                              model.params.shape[0]))
                final_evaluation = model.evaluate(full_raw_xgrid,
                                                  full_raw_ygrid)

            else:
                # Evaluate the oversampling pixel coord 'shift',
                # aka for an oversample factor of 4, this will be
                # [-0.375 -0.125  0.125  0.375]
                oversample_shift = (torch.arange(oversample[i],
                                                 device=self.x.device) -
                                    ((oversample[i]-1)/2)) * (1/oversample[i])

                # Make oversampled sub-pixel coord grid
                oversampled_x_tiles = torch.tile(self.base_xgrid[curr_mask],
                                                 (oversample[i], 1)).T
                oversampled_x_coo = (oversampled_x_tiles + oversample_shift)
                oversampled_y_tiles = torch.tile(self.base_ygrid[curr_mask],
                                                 (oversample[i], 1)).T
                oversampled_y_coo = (oversampled_y_tiles + oversample_shift)

                # Manually meshgrid subpixel coords
                oversampled_xgrid = torch.tile(oversampled_x_coo,
                                               (1, oversample[i]))
                oversampled_ygrid = torch.repeat_interleave(oversampled_y_coo,
                                                            oversample[i],
                                                            axis=1)
                full_xgrid = torch.stack(([oversampled_xgrid] *
                                          model.params.shape[0]))
                full_ygrid = torch.stack(([oversampled_ygrid] *
                                          model.params.shape[0]))

                '''
                # Plot the sub-pixel coord grid
                plt.scatter(oversampled_xgrid, oversampled_ygrid, marker='+')
                for k in range(len(self.x)):
                    plt.axvline(k+0.5,0,1, color='black', lw=1)
                    plt.axhline(k+0.5,0,1, color='black', lw=1)
                plt.show()
                '''

                # Evaluate over sub-pixel grid
                evaluation = model.evaluate(full_xgrid, full_ygrid)

                # Downsample the evaluated grid to the pixel scale.
                final_evaluation = (torch.sum(evaluation, dim=2)
                                    / oversample[i]**2)

            curr_mask = torch.broadcast_to(curr_mask, (model.params.shape[0],
                                                       *curr_mask.shape))
            model_image[curr_mask] = final_evaluation.flatten()

        # Evaluate any pixels outside specified radii, if any
        exp_oversample_radii = oversample_radii.copy()
        exp_oversample_radii.insert(0, 0)
        remaining_mask = (centred_base_xgrid**2 + centred_base_ygrid**2 >
                          exp_oversample_radii[-1]**2)
        if remaining_mask.any():
            remaining_xgrid = self.base_xgrid[remaining_mask].unsqueeze(-1)
            remaining_ygrid = self.base_ygrid[remaining_mask].unsqueeze(-1)
            full_xgrid = torch.stack(([remaining_xgrid]*model.params.shape[0]))
            full_ygrid = torch.stack(([remaining_ygrid]*model.params.shape[0]))
            remaining_evaluation = model.evaluate(full_xgrid, full_ygrid)
            remaining_mask = torch.broadcast_to(remaining_mask,
                                                (model.params.shape[0],
                                                 *remaining_mask.shape))
            model_image[remaining_mask] = remaining_evaluation.flatten()

        return model_image

    def PSFconvolve(self, model_image, psf):
        """ PSF convolution of an image cube using fast fourier transforms. """

        # Take fast fourier transform of model image
        img_fft = torch.fft.rfft2(model_image)

        # Pad the PSF to match image shape with origin at image_shape // 2,
        # with thanks to https://stackoverflow.com/questions/54877892/
        sz = model_image.shape[1:]  # the sizes we're matching
        psf_shape = psf.shape[1:]
        sz = (sz[0] - psf_shape[0], sz[1] - psf_shape[1])
        psf = torch.nn.functional.pad(psf, ((sz[0]+1)//2, sz[0]//2,
                                            (sz[1]+1)//2, sz[1]//2),
                                      'constant')

        # Shift the PSF image origin to the top left, required for Fourier
        psf = torch.fft.ifftshift(psf, dim=(-2, -1))

        # Take fast fourier transform of PSF image
        psf_fft = torch.fft.rfft2(psf, s=model_image.shape[1:])

        # Convolve image and psf, then inverse fourier transform
        conv_fft = img_fft * psf_fft

        conv_im = torch.fft.irfft2(conv_fft, s=model_image.shape[1:])

        return conv_im
