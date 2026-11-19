import torch
import numpy as np
import os
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

torch.set_default_dtype(torch.float64)
install_dir = os.path.dirname(os.path.realpath(__file__))

# For numerical stability
device = torch.device('cpu')


def make_oversampling_table(ImageModel, Sersic, nref=1000):
    """ Determine the minimum oversampling necessary for pixel fractional
    errors less than 0.001. """

    print(f"Generating oversampling table on {device}.")

    # Grid in 'r_eff' and 'n' parameter space
    n_array = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                            6.0, 7.0, 8.0, 9.0, 10.0])
    r_arr = torch.tensor([0.1, 0.25, 0.5, 0.75, 1., 2., 3., 4., 5., 6., 8.,
                          10., 12., 14., 16., 18., 20.])

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
            if os.path.isfile(install_dir + f'/reference_images_{nref}/'
                              f'{r_arr[i]}_{n_array[j]}.npy'):
                print('Loading savefile at ' + install_dir +
                      f'/reference_images_{nref}/'
                      f'{r_arr[i]}_{n_array[j]}.npy')
                reference_image = np.load(install_dir + '/reference_images_'
                                          f'{nref}/{r_arr[i]}_{n_array[j]}.npy'
                                          ).astype(np.float64)
                np.save(install_dir + f'/reference_images_{nref}/{r_arr[i]}_'
                        f'{n_array[j]}.npy', reference_image)
                reference_image = torch.tensor(reference_image)

            else:
                print(f'Generating (Reff,n) = {r_arr[i]}, {n_array[j]}')
                # Use chunks to avoid OoM error
                ref_oversample = nref
                chunks = 10
                reference_image = torch.zeros(1, 101, 101)
                chx_orig = torch.cat((torch.arange(chunks) * (101//chunks),
                                      torch.tensor([101])))
                chy_orig = torch.cat((torch.arange(chunks) * (101//chunks),
                                      torch.tensor([101])))
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
                        refpars = refpars.unsqueeze(0)
                        refmodel = ImageModel(torch.arange(nx),
                                              torch.arange(ny),
                                              [Sersic()],
                                              None,
                                              oversample=ref_oversample,
                                              device=device,
                                              dt=torch.float32)
                        refmodel.update_parameters(refpars, torch.tensor([0]))
                        refimage = refmodel.render().cpu()
                        refimage = refimage.to(dtype=torch.get_default_dtype())
                        reference_image[:,
                                        chy_orig[chy]:
                                        chy_orig[chy+1],
                                        chx_orig[chx]:
                                        chx_orig[chx+1]] = refimage

                np.save(install_dir + f'/reference_images_{nref}/{r_arr[i]}_'
                        f'{n_array[j]}.npy', reference_image.numpy())

            """Determine optimum radii."""
            # Make ignorant image
            model = ImageModel(torch.arange(101),
                               torch.arange(101),
                               [Sersic()],
                               None,
                               device=device,
                               dt=torch.float32)
            modpars = torch.tensor([0., r_arr[i], n_array[j],
                                    50, 50, 0., 0.])
            modpars = modpars.unsqueeze(0)
            model.update_parameters(modpars, torch.tensor([0]))
            model.update_oversampling(oversample=None)
            ignorant_image = model.render().cpu()
            ignorant_image = ignorant_image.to(dtype=torch.get_default_dtype())
            residual = torch.abs(reference_image - ignorant_image)

            # Find best radii
            total_residual = residual.sum()
            base_xgrid, base_ygrid = torch.meshgrid(torch.arange(101),
                                                    torch.arange(101),
                                                    indexing='xy')
            centred_base_xgrid = base_xgrid - 50
            centred_base_ygrid = base_ygrid - 50

            log_contained = 0.
            opt_r = [0., 0., 0.]

            for r in torch.linspace(1., 101, 100000):
                curr_mask = (centred_base_xgrid**2 +
                             centred_base_ygrid**2 <= r**2)
                curr_residual = residual[:, curr_mask].sum()
                contained = curr_residual/total_residual

                # If central region contains greater than one third of
                # the residual volume
                if (contained >= 0.333):
                    log_contained = contained
                    opt_r[0] = r.item()
                    break
                if r == 101:
                    raise Exception('Contained residual criterion '
                                    'never fulfilled.')

            for r in torch.linspace(opt_r[0]+1., 101, 100000):
                curr_mask = (centred_base_xgrid**2 +
                             centred_base_ygrid**2 <= r**2)
                curr_residual = residual[:, curr_mask].sum()
                contained = curr_residual/total_residual

                # If first annulus contains half of the residual volume
                # left over from the central region
                if (((log_contained == 1.) |
                     (contained >= log_contained + 0.5*(1.-log_contained)))):
                    log_contained = contained
                    opt_r[1] = r.item()
                    break
                if r == 101:
                    raise Exception('Contained residual criterion '
                                    'never fulfilled.')

            for r in torch.linspace(opt_r[1]+1., 101, 100000):
                curr_mask = (centred_base_xgrid**2 +
                             centred_base_ygrid**2 <= r**2)
                curr_residual = residual[:, curr_mask].sum()
                contained = curr_residual/total_residual

                # If second annuli contains 99% of the residual volume
                # left over from the central-region + first-annulus
                if (((log_contained == 1.) |
                     (contained >= log_contained + 0.999*(1.-log_contained)))):
                    opt_r[2] = r.item()
                    break
                if r == 101:
                    raise Exception('Contained residual criterion '
                                    'never fulfilled.')

            print(opt_r)
            radii_table[i, j] = opt_r

            '''
            fig, ax = plt.subplots()
            ax.imshow(np.log10(residual[0]))
            for r in opt_r:
                pat = Circle((50, 50), r, facecolor='none', edgecolor='red')
                ax.add_patch(pat)
            ax.set_title(f'Residuals ($R_{{eff}}={r_arr[i]:.1f}$, '
                         f'$n={n_array[j]:.1f}$)')
            plt.show()
            '''

            """Determine optimum oversampling."""
            # Initiate starting factor and radii values
            opt_sam = [1, 1, 1]

            # Loop over radii
            for k in range(3):

                # While the maximum residual is greater than one 1000th the
                # maximum reference image value.
                while True:

                    model.update_oversampling(oversample=opt_sam,
                                              oversample_radii=opt_r)
                    image = model.render().cpu()
                    image = image.to(dtype=torch.get_default_dtype())

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

                    print(opt_sam)
                    print(torch.max(error[0][curr_mask]))

                    # If criterion reached or maxed out, break
                    cond = ((torch.max(error[0][curr_mask]) < 0.001) |
                            (opt_sam[k] == nref))

                    if cond:
                        break

                    else:
                        opt_sam[k] += 1
                        continue

            print(opt_sam)
            oversample_table[i, j] = opt_sam

            globcount += 1

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
            hdul.writeto(install_dir + f'/automatic_oversampling_{nref}.fits',
                         overwrite=True)
