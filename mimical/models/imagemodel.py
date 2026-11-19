from .rotationmodels import Rotator
import matplotlib.pyplot as plt
import torch
torch.set_default_dtype(torch.float64)


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

    oversample : int or list of ints
        The factor by which to oversample the image. Integer for 'homogeneous'
        and 'window' methods, list for 'annuli' method.

    oversample_bl : int
        Length of centred square in which to oversample.

    oversample_radii : list of floats
        Radii of annuli in which to oversample.
    """

    def __init__(self, x, y, submodels, psf, oversample=None,
                 oversample_bl=None, oversample_radii=None,
                 rotmethod='interpolation', device=torch.device('cpu'),
                 dt=torch.float32):

        self.device = device
        self.dt = dt

        self.x = x.to(self.dt).to(self.device)
        self.y = y.to(self.dt).to(self.device)
        self.nsources = [i.params.shape[1] for i in submodels]
        self.submodels = submodels

        # Cache PSF coord grid for efficiency
        if psf is not None:
            self.psf = psf.to(self.dt).to(self.device)
            psfx = torch.arange(psf[0].shape[1],
                                device=self.device,
                                dtype=self.dt)
            psfy = torch.arange(psf[0].shape[0],
                                device=self.device,
                                dtype=self.dt)
            self.psf_xgrid, self.psf_ygrid = torch.meshgrid(psfx, psfy,
                                                            indexing='xy')
            self.utilnum = torch.arange(self.psf.shape[0],
                                        device=self.device,
                                        dtype=self.dt)
        else:
            self.psf = psf

        self.oversample = oversample
        self.oversample_bl = oversample_bl
        self.oversample_radii = oversample_radii

        # Cache image coord grid for efficiency
        self.base_xgrid, self.base_ygrid = torch.meshgrid(self.x,
                                                          self.y,
                                                          indexing='xy')

        if rotmethod == 'interpolation':
            self.rotator = Rotator(device=self.device,
                                   dt=self.dt).interpolation
        elif rotmethod == 'intersection':
            self.rotator = Rotator(device=self.device,
                                   dt=self.dt).intersection
        else:
            raise Exception("Can only choose 'interpolation', or "
                            "'intersection' for rotation method.")

    def update_parameters(self, submodel_params, psf_pa):
        """ Update the submodel parameters and psf position angles. """

        submodel_params = submodel_params.to(self.dt).to(self.device)
        psf_pa = psf_pa.to(self.dt).to(self.device)

        for i in range(len(self.nsources)):
            if i == 0:
                pars = submodel_params[:, :self.nsources[i]]
                self.submodels[i].update_parameters(pars)
            else:
                partrack = sum(self.nsources[:i])
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
                psf_rot = self.rotator(self.psf, self.psf_pa,
                                       base_x=self.psf_xgrid,
                                       base_y=self.psf_ygrid,
                                       utilnum=self.utilnum)
                norm = 1 / torch.sum(psf_rot, (1, 2))
                psf_rot = psf_rot * norm[:, None, None]

            # Convolve submodel image with PSF image
            out = self.PSFconvolve(model_image, psf_rot)

        return out

    def evaluate_over_grid(self, model, oversample, oversample_bl,
                           oversample_radii):
        """ Evaluate submodel over the pixel grid. """

        # If no oversampling specified
        if oversample is None:
            # Make pixel grid
            xgrid = self.base_xgrid.unsqueeze(0).expand(model.params.shape[0],
                                                        -1, -1)
            ygrid = self.base_ygrid.unsqueeze(0).expand(model.params.shape[0],
                                                        -1, -1)
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

        oversample_shift = (torch.arange(oversample,
                                         device=self.device,
                                         dtype=self.dt)
                            - ((oversample-1) / 2)) * (1 / oversample)

        shift_x, shift_y = torch.meshgrid(oversample_shift,
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
                                  device=self.device,
                                  dtype=self.dt)

        # Inside box
        curr_mask = self.base_xgrid != self.base_xgrid
        curr_mask[(self.y.shape[0]-oversample_bl)//2:
                  -(self.y.shape[0]-oversample_bl)//2,
                  (self.x.shape[0]-oversample_bl)//2:
                  -(self.x.shape[0]-oversample_bl)//2] = True

        oversample_shift = (torch.arange(oversample,
                                         device=self.device,
                                         dtype=self.dt)
                            - ((oversample-1) / 2)) * (1 / oversample)

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
        model_image = torch.zeros((model.params.shape[0],
                                   *self.base_xgrid.shape),
                                  device=self.device,
                                  dtype=self.dt)

        # Calculate radius for each image using its own centroid
        centred_base_xgrid = (self.base_xgrid[None, :, :] -
                              self.submodels[0].x_0.reshape(-1, 1, 1))
        centred_base_ygrid = (self.base_ygrid[None, :, :] -
                              self.submodels[0].y_0.reshape(-1, 1, 1))

        r2 = centred_base_xgrid.square() + centred_base_ygrid.square()

        buckets = torch.tensor([r*r for r in oversample_radii],
                               device=self.device, dtype=self.dt)

        bucketsamp = [*oversample, 1]

        # Find required oversampling factor for each image/pixel
        annulus = torch.bucketize(r2, buckets)
        factors = torch.tensor(bucketsamp, device=self.device)
        shared_factors = factors[annulus].max(dim=0).values

        # Loop over oversampling radii
        for n in sorted(set(bucketsamp)):
            curr_mask = shared_factors == n

            # If oversample is 1, skip.
            if n == 1:
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
                oversample_shift = ((torch.arange(n,
                                                  device=self.device,
                                                  dtype=self.dt)
                                    - (n - 1) / 2) / n)

                shift_x, shift_y = torch.meshgrid(oversample_shift,
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

        # Pad to avoid wrapping around image edges
        H, W = model_image.shape[-2:]
        ph, pw = psf.shape[-2:]
        fft_shape = (H + ph - 1, W + pw - 1)

        # Fourier transform both the image and PSF
        img_fft = torch.fft.rfft2(model_image, s=fft_shape)
        # Don't shift the PSF origin, just crop the result
        psf_fft = torch.fft.rfft2(psf, s=fft_shape)

        # Convolve and inverse fourier transform
        conv_im = torch.fft.irfft2(img_fft * psf_fft, s=fft_shape)

        # Crop the shifted image
        return conv_im[...,
                       ph//2:ph//2+H,
                       pw//2:pw//2+W]
