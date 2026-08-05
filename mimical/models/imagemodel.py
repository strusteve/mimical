from .rotationmodels import Rotator
import time
import torch
import warnings
import matplotlib.pyplot as plt
import gc
# Filter out the specific dynamic resize warning
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="An output with one or more elements was resized")


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

    oversample_bl : int
        Length of centred square in which to oversample.

    oversample_radii : list of floats
        Radii of annuli in which to oversample.
    """

    def __init__(self, x, y, submodels, psf, psf_pa, oversample=None,
                 oversample_bl=None, oversample_radii=None):

        self.x = x
        self.y = y
        self.nsources = [i.params.shape[1] for i in submodels]
        self.submodels = submodels
        self.psf = psf
        self.oversample = oversample
        self.oversample_bl = oversample_bl
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

    def update_oversampling(self, oversample=None, oversample_bl=None,
                            oversample_radii=None):
        """ Update the oversampling parameters. """

        self.oversample = oversample
        self.oversample_bl = oversample_bl
        self.oversample_radii = oversample_radii

    def render(self):
        """ Render the ImageModel onto its pixel grid. """
        # Evaluate submodel over the pixel grid
        model_image = self.evaluate_over_grid(self.submodels[0],
                                              self.oversample,
                                              self.oversample_bl,
                                              self.oversample_radii)

        # For multiple sources in scene
        if len(self.nsources) > 1:
            for i in range(1, len(self.nsources)):
                newsource = self.evaluate_over_grid(self.submodels[i],
                                                    self.oversample,
                                                    self.oversample_bl,
                                                    self.oversample_radii)
                model_image += newsource

        # If no PSF is provided, return base model
        if self.psf is None:
            out = model_image

        # If PSF is provided, rotate if desired then convolve
        else:
            # Rotation
            if (self.psf_pa == 0).all():
                psf_rot = self.psf
            else:
                '''
                psf_rot = self.rot.intersection(self.psf, self.psf_pa,
                                                base_x=self.psf_xgrid,
                                                base_y=self.psf_ygrid,
                                                utilnum=self.utilnum)
                '''
                psf_rot = self.rot.interpolation(self.psf, self.psf_pa)
                norm = 1 / torch.sum(psf_rot, (1, 2))
                psf_rot = psf_rot * norm[:, None, None]

            # Convolve submodel image with PSF image
            out = self.PSFconvolve(model_image, psf_rot)

        if self.x.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.x.device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        return out

    def evaluate_over_grid(self, model, oversample, oversample_bl,
                           oversample_radii):
        """ Evaluate submodel over the pixel grid. """

        # If no oversampling specified
        if oversample is None:
            # Make pixel grid
            xgrid = torch.stack(([self.base_xgrid]*model.params.shape[0]))
            ygrid = torch.stack(([self.base_ygrid]*model.params.shape[0]))
            return model.evaluate(xgrid, ygrid)

        # Expand the pixel grid and then block-reduce
        elif (isinstance(oversample, (int, float)) &
              (oversample_bl is None) &
              (oversample_radii is None)):
            return self.homogeneous_oversampling(model, oversample)

        # Expand the window pixel grid and then block-reduce
        elif (isinstance(oversample, (int, float)) &
              isinstance(oversample_bl, (int, float))
              & (oversample_radii is None)):
            return self.window_oversampling(model, oversample,
                                            oversample_bl)

        # For inhomogeneous oversampling, loop over annuli about image centre
        elif (isinstance(oversample, list) &
              isinstance(oversample_radii, list) &
              (oversample_bl is None)):
            return self.annuli_oversampling(model, oversample,
                                            oversample_radii)

        else:
            raise Exception("Invalid syntax")

    def homogeneous_oversampling(self, model, oversample):
        """ Oversample the entire image by the factor provided. """

        oversample_shift = (torch.arange(oversample, device=self.x.device) -
                            ((oversample-1) / 2)) * (1 / oversample)

        shift_x, shift_y = torch.meshgrid(
            oversample_shift,
            oversample_shift,
            indexing="xy")

        xgrid = self.base_xgrid.flatten()
        ygrid = self.base_ygrid.flatten()

        xgrid = xgrid[:, None, None] + shift_x
        ygrid = ygrid[:, None, None] + shift_y

        xgrid = xgrid.reshape(xgrid.shape[0], -1)
        ygrid = ygrid.reshape(ygrid.shape[0], -1)

        xgrid = xgrid.unsqueeze(0).expand(model.params.shape[0], -1, -1)
        ygrid = ygrid.unsqueeze(0).expand(model.params.shape[0], -1, -1)

        evaluation = model.evaluate(xgrid, ygrid)
        evaluation = torch.mean(evaluation, dim=2)

        return evaluation.reshape(model.params.shape[0],
                                  len(self.y),
                                  len(self.x))

    def window_oversampling(self, model, oversample, oversample_bl):
        """ Oversampled a centred window of length 'oversample_bl'. """

        model_image = torch.zeros((model.params.shape[0],
                                   *self.base_xgrid.shape),
                                  device=self.base_xgrid.device,
                                  dtype=torch.float32)

        # Inside box
        curr_mask = self.base_xgrid != self.base_xgrid
        curr_mask[(self.y.shape[0]-oversample_bl)//2:
                  -(self.y.shape[0]-oversample_bl)//2,
                  (self.x.shape[0]-oversample_bl)//2:
                  -(self.x.shape[0]-oversample_bl)//2] = True

        oversample_shift = (torch.arange(oversample, device=self.x.device) -
                            ((oversample-1) / 2)) * (1 / oversample)

        shift_x, shift_y = torch.meshgrid(oversample_shift,
                                          oversample_shift,
                                          indexing="xy")

        xgrid = self.base_xgrid[curr_mask]
        ygrid = self.base_ygrid[curr_mask]

        xgrid = xgrid[:, None, None] + shift_x
        ygrid = ygrid[:, None, None] + shift_y

        xgrid = xgrid.reshape(xgrid.shape[0], -1)
        ygrid = ygrid.reshape(ygrid.shape[0], -1)

        xgrid = xgrid.unsqueeze(0).expand(model.params.shape[0], -1, -1)
        ygrid = ygrid.unsqueeze(0).expand(model.params.shape[0], -1, -1)

        # Evaluate and downsample
        evaluation = model.evaluate(xgrid, ygrid)
        evaluation = torch.mean(evaluation, dim=2)
        model_image[:, curr_mask] = evaluation

        # Outside box
        xgrid = self.base_xgrid[~curr_mask].unsqueeze(-1)
        ygrid = self.base_ygrid[~curr_mask].unsqueeze(-1)
        xgrid = xgrid.unsqueeze(0).expand(model.params.shape[0], -1, -1)
        ygrid = ygrid.unsqueeze(0).expand(model.params.shape[0], -1, -1)
        evaluation = model.evaluate(xgrid, ygrid).squeeze(-1)
        model_image[:, ~curr_mask] = evaluation

        return model_image

    def annuli_oversampling(self, model, oversample, oversample_radii):
        """ Oversampling in annulii by the factors 'oversample'. """

        # Make a centred coordinate grid
        model_image = torch.zeros(
            (model.params.shape[0], *self.base_xgrid.shape),
            device=self.base_xgrid.device,
            dtype=torch.float32,
        )

        # centred_base_xgrid = self.base_xgrid - ((self.x.shape[0]-1) / 2)
        # centred_base_ygrid = self.base_ygrid - ((self.y.shape[0]-1) / 2)
        centred_base_xgrid = self.base_xgrid - \
            torch.mean(self.submodels[0].x_0)
        centred_base_ygrid = self.base_ygrid - \
            torch.mean(self.submodels[0].y_0)

        # Bucketize
        r2 = centred_base_xgrid.square() + centred_base_ygrid.square()
        buckets = torch.tensor([0,
                                *[r**r for r in oversample_radii],
                                1e99],
                               device=self.base_xgrid.device)
        bucketsamp = [*oversample, 1]
        annulus = torch.bucketize(r2, buckets)

        # Loop over oversampling radii
        for i in range(len(buckets)):

            curr_mask = annulus == (i+1)

            if not curr_mask.any():
                continue

            # If oversample is 1, skip.
            if bucketsamp[i] == 1:
                # Evaluate over sub-pixel grid
                xgrid = self.base_xgrid[curr_mask].unsqueeze(-1)
                ygrid = self.base_ygrid[curr_mask].unsqueeze(-1)
                xgrid = xgrid.unsqueeze(0).expand(model.params.shape[0],
                                                  -1, -1)
                ygrid = ygrid.unsqueeze(0).expand(model.params.shape[0],
                                                  -1, -1)
                evaluation = model.evaluate(xgrid, ygrid).squeeze(-1)

            else:
                # Evaluate the oversampling pixel coord 'shift',
                # aka for an oversample factor of 4, this will be
                # [-0.375 -0.125  0.125  0.375]
                n = bucketsamp[i]
                oversample_shift = (
                    (torch.arange(n, device=self.x.device) - (n - 1) / 2) / n
                )

                shift_x, shift_y = torch.meshgrid(
                    oversample_shift,
                    oversample_shift,
                    indexing="xy",)

                xgrid = self.base_xgrid[curr_mask]
                ygrid = self.base_ygrid[curr_mask]

                xgrid = xgrid[:, None, None] + shift_x
                ygrid = ygrid[:, None, None] + shift_y

                xgrid = xgrid.reshape(xgrid.shape[0], -1)
                ygrid = ygrid.reshape(ygrid.shape[0], -1)

                xgrid = xgrid.unsqueeze(0).expand(model.params.shape[0],
                                                  -1, -1)
                ygrid = ygrid.unsqueeze(0).expand(model.params.shape[0],
                                                  -1, -1)

                '''
                # Plot the sub-pixel coord grid
                plt.scatter(oversampled_xgrid, oversampled_ygrid, marker='+')
                for k in range(len(self.x)):
                    plt.axvline(k+0.5,0,1, color='black', lw=1)
                    plt.axhline(k+0.5,0,1, color='black', lw=1)
                plt.show()
                '''

                # Evaluate over sub-pixel grid
                evaluation = model.evaluate(xgrid, ygrid)
                evaluation = evaluation.mean(dim=2)

            model_image[:, curr_mask] = evaluation

        return model_image

    def PSFconvolve(self, model_image, psf):
        """ PSF convolution of an image cube using fast fourier transforms. """

        # Take fast fourier transform of model image
        # 1/3 time
        img_fft = torch.fft.rfft2(model_image)

        # Pad the PSF to match image shape with origin at image_shape // 2,
        # with thanks to https://stackoverflow.com/questions/54877892/
        sz = model_image.shape[1:]
        psf_shape = psf.shape[1:]
        sz = (sz[0] - psf_shape[0], sz[1] - psf_shape[1])
        psf = torch.nn.functional.pad(psf, ((sz[0]+1)//2, sz[0]//2,
                                            (sz[1]+1)//2, sz[1]//2),
                                      'constant')

        # Shift the PSF image origin to the top left, required for Fourier
        psf = torch.fft.ifftshift(psf, dim=(-2, -1))

        # Take fast fourier transform of PSF image
        # 1/3 time
        psf = torch.fft.rfft2(psf, s=model_image.shape[1:])

        # Convolve image and psf, then inverse fourier transform
        conv_fft = img_fft * psf

        # 1/3 time
        conv_im = torch.fft.irfft2(conv_fft, s=model_image.shape[1:])

        return conv_im
