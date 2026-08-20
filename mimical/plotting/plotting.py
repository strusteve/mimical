import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import cmcrameri.cm as cmc
from matplotlib.ticker import (AutoMinorLocator, AutoLocator)
import torch
fs = 6


def plot_best(images, wavs, image_models, best_sample, prior_handler,
              filter_names, segmaps, oversample=None,
              oversample_bl=None, oversample_radii=None):
    """ Plots the maximum likelihood model and resid."""

    # Pass segmaps through images
    images *= segmaps

    # Initiate plot
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=4, ncols=len(images))

    # Get median parameters and translate into median model parameters.
    param_dict = prior_handler.revert(best_sample)
    pars = param_dict[:, :np.sum(prior_handler.nsources)]
    psfarr = param_dict[:, np.sum(prior_handler.nsources)]
    pars = torch.tensor(pars, dtype=torch.float32,
                        device=image_models.x.device)

    # Generate best model
    image_models.update_parameters(pars,
                                   torch.tensor(psfarr, dtype=torch.float32,
                                                device=image_models.x.device))
    if oversample is not None:
        image_models.update_oversampling(oversample, oversample_bl,
                                         oversample_radii)

    models = image_models.render().cpu().numpy()
    resid = (images-models)*segmaps

    # Set vmins
    vmins = [-max([np.percentile(x.flatten(), q=99.9) for x in images]),
             -max([np.percentile(x.flatten(), q=99.9) for x in images]),
             -max([np.percentile(x.flatten(), q=99.9) for x in images]),
             min(min([np.percentile(x.flatten(), q=1) for x in resid]),
                 -max([-np.percentile(x.flatten(), q=99) for x in resid]))]

    # Set vmaxs
    vmaxs = [max([np.percentile(x.flatten(), q=99.9) for x in images]),
             max([np.percentile(x.flatten(), q=99.9) for x in images]),
             max([np.percentile(x.flatten(), q=99.9) for x in images]),
             max(-min([np.percentile(x.flatten(), q=1) for x in resid]),
                 max([-np.percentile(x.flatten(), q=99) for x in resid]))]

    # Loop over filters and plot
    for i in range(len(wavs)):

        plotims = [images[i], models[i], resid[i], resid[i]]

        for j in range(4):

            ax = fig.add_subplot(gs[j, i])
            im = ax.pcolormesh(plotims[j], vmax=vmaxs[j], vmin=vmins[j],
                               cmap=cmc.managua_r, rasterized=True)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_axis_off()

            if j == 0:
                ax.text(0.95, 0.95, filter_names[i].upper(), fontsize=fs,
                        transform=ax.transAxes, ha='right', va='top',
                        color='white')

            if i == 0:
                if j == 0:
                    ax.text(0.05, 0.05, 'Data', fontsize=fs,
                            transform=ax.transAxes, ha='left', va='bottom',
                            color='white')
                if j == 1:
                    ax.text(0.05, 0.05, 'Model', fontsize=fs,
                            transform=ax.transAxes, ha='left', va='bottom',
                            color='white')
                if j == 2:
                    ax.text(0.05, 0.05, 'Residual', fontsize=fs,
                            transform=ax.transAxes, ha='left', va='bottom',
                            color='white')
                if j == 3:
                    ax.text(0.05, 0.05, 'Residual\nZoom', fontsize=fs,
                            transform=ax.transAxes, ha='left', va='bottom',
                            color='white')

    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    fig.set_size_inches(len(images), 4, forward=True)


def plot_trends(wavs, samples, mimical_prior, prior_handler, mimical_keys):
    """ Plot the 2D model parameter relationship with filter wavelength. """

    # Convert multi-band samples into model parameter samples
    samples_mimical = np.apply_along_axis(lambda samp:
                                          prior_handler.revert(samp).flatten(),
                                          1, samples)
    samples_mimical = samples_mimical.reshape(samples.shape[0], len(wavs),
                                              len(mimical_keys))

    # Get model parameter posterior stats
    stats = np.percentile(samples_mimical, axis=0,
                          q=(0.15, 2.5, 16, 50, 84, 97.5, 99.85))

    # Loop over model parameters
    fig, ax = plt.subplots(len(mimical_keys)//3+len(mimical_keys) % 3, 3)

    axc = 0
    for i in range(len(list(mimical_prior.keys()))):

        priorkey = list(mimical_prior.keys())[i]

        if 'source' in priorkey:

            for j in range(len(list(mimical_prior[priorkey].keys()))):

                subkey = list(mimical_prior[priorkey].keys())[j]

                ax.flatten()[axc].fill_between(wavs, stats[0].T[axc],
                                               stats[6].T[axc], color='black',
                                               alpha=0.05, lw=0)
                ax.flatten()[axc].fill_between(wavs, stats[1].T[axc],
                                               stats[5].T[axc], color='black',
                                               alpha=0.1, lw=0)
                ax.flatten()[axc].fill_between(wavs, stats[2].T[axc],
                                               stats[4].T[axc], color='black',
                                               alpha=0.25, lw=0)
                ax.flatten()[axc].plot(wavs, stats[3].T[axc], color='black')
                ax.flatten()[axc].set_ylabel(mimical_keys[axc])
                ax.flatten()[axc].set_xlabel('$\\lambda$')

                labstr = ''
                if mimical_prior[priorkey][subkey][1] == 'Individual':
                    labstr += 'Individual'
                else:
                    labstr += f'{mimical_prior[priorkey][subkey][1]}, ' +\
                              f'{mimical_prior[priorkey][subkey][2]}'

                if isinstance(mimical_prior[priorkey][subkey][0], tuple):
                    labstr += '\nFitted'
                elif isinstance(mimical_prior[priorkey][subkey][0], str):
                    if mimical_prior[priorkey][subkey][0] == 'Infer':
                        labstr += '\nInferred'
                else:
                    labstr += '\nFixed'

                ax.flatten()[axc].text(0.95, 0.95, labstr, fontsize=fs,
                                       transform=ax.flatten()[axc].transAxes,
                                       ha='right', va='top', color='black',
                                       bbox=dict(boxstyle='round',
                                                 facecolor='white',
                                                 alpha=0.75))
                ax.flatten()[axc].xaxis.set_major_locator(AutoLocator())
                ax.flatten()[axc].xaxis.set_minor_locator(AutoMinorLocator())
                ax.flatten()[axc].yaxis.set_major_locator(AutoLocator())
                ax.flatten()[axc].yaxis.set_minor_locator(AutoMinorLocator())
                ax.flatten()[axc].tick_params(which='both', direction='in',
                                              top=True, right=True)

                axc += 1

        else:

            ax.flatten()[axc].fill_between(wavs, stats[0].T[axc],
                                           stats[6].T[axc], color='black',
                                           alpha=0.05, lw=0)
            ax.flatten()[axc].fill_between(wavs, stats[1].T[axc],
                                           stats[5].T[axc], color='black',
                                           alpha=0.1, lw=0)
            ax.flatten()[axc].fill_between(wavs, stats[2].T[axc],
                                           stats[4].T[axc], color='black',
                                           alpha=0.25, lw=0)
            ax.flatten()[axc].plot(wavs, stats[3].T[axc], color='black')
            ax.flatten()[axc].set_ylabel(mimical_keys[axc])
            ax.flatten()[axc].set_xlabel('$\\lambda$')

            labstr = ''
            if mimical_prior[priorkey][1] == 'Individual':
                labstr += 'Individual'
            else:
                labstr += f'{mimical_prior[priorkey][1]}, ' +\
                          f'{mimical_prior[priorkey][2]}'
            if isinstance(mimical_prior[priorkey][0], tuple):
                labstr += '\nFitted'
            elif isinstance(mimical_prior[priorkey][0], str):
                if mimical_prior[priorkey][0] == 'Infer':
                    labstr += '\nInferred'
            else:
                labstr += '\nFixed'

            ax.flatten()[axc].text(0.95, 0.95, labstr, fontsize=fs,
                                   transform=ax.flatten()[axc].transAxes,
                                   ha='right', va='top', color='black',
                                   bbox=dict(boxstyle='round',
                                             facecolor='white',
                                             alpha=0.75))
            ax.flatten()[axc].xaxis.set_major_locator(AutoLocator())
            ax.flatten()[axc].xaxis.set_minor_locator(AutoMinorLocator())
            ax.flatten()[axc].yaxis.set_major_locator(AutoLocator())
            ax.flatten()[axc].yaxis.set_minor_locator(AutoMinorLocator())
            ax.flatten()[axc].tick_params(which='both', direction='in',
                                          top=True, right=True)

            axc += 1

    for k in range(len(ax.flatten())-axc):
        ax.flatten()[-(k+1)].set_axis_off()

    fig.set_size_inches(7.03058,
                        ((len(mimical_keys)//3+len(mimical_keys) % 3)/3)
                        * 7.03058)
    fig.tight_layout()


def plot_errors(images, wavs, mimical_prior, image_models, best_sample,
                prior_handler, filter_names, segmaps, oversample=None,
                oversample_bl=None, oversample_radii=None):
    """ Provides a summary of the individual errors present in best model. """

    # Initiate plot
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=4, ncols=len(images)+1,
                          width_ratios=np.append(np.ones(len(images)), 0.25))

    # Translate best sample into per-filter model parameters.
    param_dict = best_sample
    reverted = prior_handler.revert(param_dict)
    pars = reverted[:, :np.sum(prior_handler.nsources)]
    psfarr = reverted[:, np.sum(prior_handler.nsources)]
    rmsarr = reverted[:, np.sum(prior_handler.nsources)+1]
    cpfarr = reverted[:, np.sum(prior_handler.nsources)+2]

    # If user provides RMS values, override prior sample
    if isinstance(mimical_prior['rms'][0], list):
        if isinstance(mimical_prior['rms'][0][0], np.ndarray):
            rmsarr = np.array(mimical_prior['rms'][0])

    # If user provides counts-per-flux parameters, override prior sample
    if isinstance(mimical_prior['counts_per_flux'][0], list):
        if isinstance(mimical_prior['counts_per_flux'][0][0], np.ndarray):
            cpfarr = np.array(mimical_prior['counts_per_flux'][0])

    image_models.update_parameters(torch.tensor(pars, dtype=torch.float32,
                                                device=image_models.x.device),
                                   torch.tensor(psfarr, dtype=torch.float32,
                                                device=image_models.x.device))
    if oversample is not None:
        image_models.update_oversampling(oversample, oversample_bl,
                                         oversample_radii)

    models = image_models.render().cpu().numpy()

    # Create master lists for appending
    rmserr = (np.zeros_like(images.T) + rmsarr.T).T
    poissonerr = ((cpfarr.T**(-1/2))*np.sqrt(np.abs(models.T))).T
    sigmaerr = np.sqrt(rmserr**2 + (poissonerr)**2)
    ratio = (poissonerr/(poissonerr+rmserr))*100

    # Set vmins
    vmins = [-np.max(([x.max() for x in sigmaerr])),
             -np.max(([x.max() for x in sigmaerr])),
             -np.max(([x.max() for x in sigmaerr])),
             0]

    # Set vmaxs
    vmaxs = [np.max(([x.max() for x in sigmaerr])),
             np.max(([x.max() for x in sigmaerr])),
             np.max(([x.max() for x in sigmaerr])),
             100]

    # Initiate colorbars
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    im1 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[0],
                        vmin=vmins[0], cmap=cmc.managua_r,
                        rasterized=True)
    cbarax1 = fig.add_subplot(gs[:3, -1])
    cbarax1.set_yticks([])
    cbarax1.set_xticks([])
    cbar1 = plt.colorbar(im1, cax=cbarax1, fraction=1)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbar1.locator = tick_locator
    cbar1.update_ticks()
    im2 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[-1],
                        vmin=vmins[-1], cmap=cmc.managua_r,
                        rasterized=True)
    cbarax2 = fig.add_subplot(gs[3, -1])
    cbarax2.set_yticks([])
    cbarax2.set_xticks([])
    cbar2 = plt.colorbar(im2, cax=cbarax2, fraction=1)
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar2.locator = tick_locator
    cbar2.update_ticks()

    # Loop over filters and plot
    for i in range(len(wavs)):

        plotims = [rmserr[i], poissonerr[i], sigmaerr[i], ratio[i]]

        for j in range(4):

            ax = fig.add_subplot(gs[j, i])
            im = ax.pcolormesh(plotims[j]*segmaps[i], vmax=vmaxs[j],
                               vmin=vmins[j], cmap=cmc.managua_r,
                               rasterized=True)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_axis_off()

            if j == 0:
                ax.text(0.95, 0.95, filter_names[i].upper(), fontsize=fs,
                        transform=ax.transAxes, ha='right', va='top',
                        color='white')

            if i == 0:
                if j == 0:
                    ax.text(0.05, 0.05, 'RMS', fontsize=fs,
                            transform=ax.transAxes, ha='left', va='bottom',
                            color='white')
                if j == 1:
                    ax.text(0.05, 0.05, 'Poisson', fontsize=fs,
                            transform=ax.transAxes, ha='left', va='bottom',
                            color='white')

                if j == 2:
                    ax.text(0.05, 0.05, 'Total Sigma', fontsize=fs,
                            transform=ax.transAxes, ha='left', va='bottom',
                            color='white')

                if j == 3:
                    ax.text(0.05, 0.05, '%Poisson', fontsize=fs,
                            transform=ax.transAxes, ha='left', va='bottom',
                            color='white')

    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    fig.set_size_inches(len(images), 4, forward=True)
