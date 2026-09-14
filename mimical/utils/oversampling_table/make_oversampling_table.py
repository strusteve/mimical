import torch
import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

install_dir = os.path.dirname(os.path.realpath(__file__))
device = (torch.accelerator.current_accelerator()
          if torch.accelerator.is_available()
          else torch.device('cpu'))


def make_oversampling_table(ImageModel, Sersic):
    """ Determine the minimum oversampling necessary for pixel fractional
    errors less than 0.001. """

    # Grid in 'r_eff' and 'n' parameter space
    n_array = torch.tensor([0.1, 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.,
                            5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.])
    r_arr = torch.tensor([0.1, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.,
                          12., 13., 14., 15., 16., 17., 18., 19., 20.])

    # Table to append oversampling factors to
    oversample_table = np.zeros((len(r_arr), len(n_array), 3), dtype='int')
    radii_table = np.zeros((len(r_arr), len(n_array), 3), dtype='float')

    globcount = 0
    # Loop over 'r_eff' and 'n' parameter space
    print('One-time generation of oversampling table...')
    for i in range(len(r_arr)):
        for j in range(len(n_array)):

            print(f'{globcount / (len(r_arr)*len(n_array))*100:.1f}%')

            """Make 'true' reference image."""
            if os.path.isfile(install_dir + '/reference_images/'
                              f'{r_arr[i]}_{n_array[j]}.npy'):
                print('present')
                reference_image = np.load(install_dir + '/reference_images/'
                                          f'{r_arr[i]}_{n_array[j]}.npy')
                reference_image = torch.tensor(reference_image)

            else:
                # Use chunks to avoid OoM error
                ref_oversample = 1000
                chunks = 50
                reference_image = torch.zeros(1, 101, 101)
                chx_orig = torch.cat((torch.arange(chunks) * (101//chunks),
                                      torch.tensor([101]))).to(device=device)
                chy_orig = torch.cat((torch.arange(chunks) * (101//chunks),
                                      torch.tensor([101]))).to(device=device)
                for chy in range(chunks):
                    for chx in range(chunks):

                        if chx != chunks-1:
                            nx = 101 // (chunks)
                        else:
                            nx = 101 - ((101 // (chunks))*(chunks-1))
                        if chy != chunks-1:
                            ny = 101 // (chunks)
                        else:
                            ny = 101 - ((101 // (chunks))*(chunks-1))
                        ch_x_0 = 50 - chx_orig[chx]
                        ch_y_0 = 50 - chy_orig[chy]
                        refpars = torch.tensor([0, r_arr[i], n_array[j],
                                                ch_x_0, ch_y_0, 0., 0.])
                        refpars = refpars.to(torch.float32).to(device=device
                                                               ).unsqueeze(0)
                        refmodel = ImageModel(torch.arange(nx, device=device),
                                              torch.arange(ny, device=device),
                                              [Sersic()], None, [0.],
                                              oversample=ref_oversample)
                        refmodel.update_parameters(refpars, [0])
                        refimage = refmodel.render().cpu()
                        reference_image[:,
                                        chy_orig[chy]:
                                        chy_orig[chy+1],
                                        chx_orig[chx]:
                                        chx_orig[chx+1]] = refimage

                np.save(install_dir +
                        f'/reference_images/{r_arr[i]}_{n_array[j]}.npy',
                        reference_image.numpy())

            """Determine optimum radii."""
            opt_r = [1., 2., 3.]

            # Make ignorant image
            model = ImageModel(torch.arange(101, device=device),
                               torch.arange(101, device=device),
                               [Sersic()], None, [0.])
            modpars = torch.tensor([0., r_arr[i], n_array[j],
                                    50., 50., 0., 0.])
            modpars = modpars.to(torch.float32).to(device=device).unsqueeze(0)
            model.update_parameters(modpars, [0])
            model.update_oversampling(oversample=None)
            ignorant_image = model.render().cpu()
            residual = torch.abs(reference_image - ignorant_image)

            # Find best radii
            total_residual = residual.sum()
            base_xgrid, base_ygrid = torch.meshgrid(torch.arange(101),
                                                    torch.arange(101),
                                                    indexing='xy')
            centred_base_xgrid = base_xgrid - 50
            centred_base_ygrid = base_ygrid - 50

            count = 0
            contained_log = 0
            for r in torch.linspace(0, 101, 100000):
                mask = (centred_base_xgrid**2 +
                        centred_base_ygrid**2 <= r**2)
                curr_residual = residual[:, mask].sum()
                contained = curr_residual/total_residual

                # Find radii that contains first 1/3 of residual volume
                if (count == 0) & (contained > 1/3) & (r >= 1.):
                    count += 1
                    opt_r[0] = r
                    contained_log = contained

                # Find radii that contains second 1/3 of residual volume
                elif ((count == 1)
                      & (contained > contained_log + 0.5*(1-contained_log))
                      & (r >= 2.)):
                    count += 1
                    opt_r[1] = r
                    contained_log = contained

                # Find radii that contains last 1/3-0.01% of residual volume
                elif ((count == 2)
                      & (contained > contained_log + 0.99*(1-contained_log))
                      & (r >= 3.)):
                    count += 1
                    opt_r[2] = r

                elif count > 2:
                    break

            radii_table[i, j] = opt_r

            '''
            fig, ax = plt.subplots()
            ax.imshow(residual[0])
            for r in opt_r:
                pat = Circle((50, 50), r, facecolor='none', edgecolor='red')
                ax.add_patch(pat)
            ax.set_title(f'Residuals ($R_{{eff}}={r_arr[i]:.1f}$, '
                         f'$n={n_array[j]:.1f}$)')
            plt.savefig(install_dir +
                        f'/temp_plots/{r_arr[i]:.1f}_'
                        f'{n_array[j]:.1f}.pdf')
            '''

            """Determine optimum oversampling."""
            # Initiate starting factor and radii values
            opt_sam = [1, 1, 1]

            # Loop over radii
            for k in range(3):

                # While the maximum residual is greater than one 1000th the
                # maximum reference image value.
                while True:

                    print(opt_sam)

                    model.update_oversampling(oversample=opt_sam,
                                              oversample_radii=opt_r)
                    image = model.render().cpu()

                    # Calculate residual ratio w.r.t reference model
                    error = (torch.abs(reference_image - image) /
                             torch.abs(reference_image))
                    error[~torch.isfinite(error)] = 0

                    # Create mask for pixels within current radii
                    base_xgrid, base_ygrid = torch.meshgrid(torch.arange(101),
                                                            torch.arange(101),
                                                            indexing='xy')
                    centred_base_xgrid = base_xgrid - 50
                    centred_base_ygrid = base_ygrid - 50
                    # If first radii, include centre
                    if k == 0:
                        curr_mask = (centred_base_xgrid**2 +
                                     centred_base_ygrid**2 <= opt_r[k]**2)
                    # Else, mask in annuli
                    else:
                        curr_mask = ((centred_base_xgrid**2 +
                                      centred_base_ygrid**2 <= opt_r[k]**2) &
                                     (centred_base_xgrid**2 +
                                      centred_base_ygrid**2 > opt_r[k-1]**2))

                    # If no pixels in current radii, skip
                    if torch.sum(curr_mask) == 0:
                        break

                    # If criterion reached or maxed out, break
                    cond = ((torch.max(error[0][curr_mask]) < 0.01) |
                            (opt_sam[k] == 1000))
                    if cond:
                        break

                    # If not, continue
                    else:
                        opt_sam[k] += 1
                        continue

            oversample_table[i, j] = opt_sam

            """Save table."""
            radii_table1 = Table(np.column_stack([r_arr.tolist(),
                                                  radii_table[:, :, 0]]),
                                 names=["Reff\\n"] + n_array.tolist())
            radii_table2 = Table(np.column_stack([r_arr.tolist(),
                                                  radii_table[:, :, 1]]),
                                 names=["Reff\\n"] + n_array.tolist())
            radii_table3 = Table(np.column_stack([r_arr.tolist(),
                                                  radii_table[:, :, 2]]),
                                 names=["Reff\\n"] + n_array.tolist())
            hdu1 = fits.BinTableHDU(radii_table1, name="RADII_1")
            hdu2 = fits.BinTableHDU(radii_table2, name="RADII_2")
            hdu3 = fits.BinTableHDU(radii_table3, name="RADII_3")

            samp_table1 = Table(np.column_stack([r_arr.tolist(),
                                                 oversample_table[:, :, 0]]),
                                names=["Reff\\n"] + n_array.tolist())
            samp_table2 = Table(np.column_stack([r_arr.tolist(),
                                                 oversample_table[:, :, 1]]),
                                names=["Reff\\n"] + n_array.tolist())
            samp_table3 = Table(np.column_stack([r_arr.tolist(),
                                                oversample_table[:, :, 2]]),
                                names=(["Reff\\n"] + n_array.tolist()))
            hdu4 = fits.BinTableHDU(samp_table1, name="FACTOR_1")
            hdu5 = fits.BinTableHDU(samp_table2, name="FACTOR_2")
            hdu6 = fits.BinTableHDU(samp_table3, name="FACTOR_3")

            primary = fits.PrimaryHDU()
            hdul = fits.HDUList([primary, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6])
            hdul.writeto(install_dir + '/automatic_oversampling.fits',
                         overwrite=True)

            globcount += 1
