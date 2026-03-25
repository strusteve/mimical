import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution.utils import discretize_model
from tqdm import tqdm
from matplotlib import ticker
import cmcrameri.cm as cmc
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, AutoLocator)

fs = 6

class plotter(object): 
    """ Contains functions for plotting either a median 
    posterior model (plot_median) or a model based of the median of each
    individual parameter (plot_median_param).
    """


    def plot_best(self, images, wavs, convolved_models, best_sample, prior_handler, filter_names, segmaps):
        """ Plots the maximum likelihood model and residuals."""

        # Pass segmaps through images
        for i in range(len(wavs)):
            images[i] *= segmaps[i]

        # Initiate plot
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=4, ncols=len(images))
        #gs = fig.add_gridspec(nrows=4, ncols=len(images)+1, width_ratios=np.append(np.ones(len(images)), 0.25))

        # Get median Nautilus parameters and transalte into median model parameters.
        param_dict = best_sample
        pars = prior_handler.revert(param_dict)[:,:prior_handler.nmodel]

        # Create master lists for appending
        models = []
        residuals = []
        for i in range(len(wavs)):
            convolved_models[i].parameters = pars[i]
            model = discretize_model(model=convolved_models[i], 
                                    x_range=[0,images[i].shape[1]], 
                                    y_range=[0,images[i].shape[0]], 
                                    mode='center')
            models.append(model * segmaps[i])
            residuals.append((images[i] - model) * segmaps[i])

        # Set vmins
        vmins = [-max([np.percentile(x.flatten(), q=99) for x in images]), 
                 -max([np.percentile(x.flatten(), q=99) for x in images]), 
                 -max([np.percentile(x.flatten(), q=99) for x in images]), 
                 min( min([np.percentile(x.flatten(), q=1) for x in residuals]), -max([-np.percentile(x.flatten(), q=99) for x in residuals]))]
        
        # Set vmaxs
        vmaxs = [max([np.percentile(x.flatten(), q=99) for x in images]), 
                 max([np.percentile(x.flatten(), q=99) for x in images]), 
                 max([np.percentile(x.flatten(), q=99) for x in images]), 
                 max( -min([np.percentile(x.flatten(), q=1) for x in residuals]), max([-np.percentile(x.flatten(), q=99) for x in residuals]))]
        
        '''
        # Initiate colorbars
        ax = fig.add_subplot(gs[0, 0])
        ax.set_axis_off()
        im1 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[0], vmin=vmins[0], cmap=cmc.managua_r, rasterized=True)
        cbarax1 = fig.add_subplot(gs[:3, -1])
        cbarax1.set_yticks([])
        cbarax1.set_xticks([])
        cbar1 = plt.colorbar(im1, cax=cbarax1, fraction=1)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar1.locator = tick_locator
        cbar1.update_ticks()
        im2 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[-1], vmin=vmins[-1], cmap=cmc.managua_r, rasterized=True)
        cbarax2 = fig.add_subplot(gs[3, -1])
        cbarax2.set_yticks([])
        cbarax2.set_xticks([])
        cbar2 = plt.colorbar(im2, cax=cbarax2, fraction=1)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar2.locator = tick_locator
        cbar2.update_ticks()
        '''
        
        # Loop over filters and plot
        for i in range(len(wavs)):

            plotims = [images[i], models[i], residuals[i], residuals[i]]

            for j in range(4):

                ax = fig.add_subplot(gs[j, i])
                im = ax.pcolormesh(plotims[j], vmax=vmaxs[j], vmin=vmins[j], cmap=cmc.managua_r, rasterized=True)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_axis_off()

                if j==0:
                    ax.text(0.95, 0.95, filter_names[i].upper(), fontsize=fs, transform=ax.transAxes, ha='right', va='top', color='white')

                if i==0:
                    if j==0:
                        ax.text(0.05, 0.05, 'Data', fontsize=fs, transform=ax.transAxes, ha='left', va='bottom', color='white')
                    if j==1:
                        ax.text(0.05, 0.05, 'Model', fontsize=fs, transform=ax.transAxes, ha='left', va='bottom', color='white')
                    if j==2:
                        ax.text(0.05, 0.05, 'Residual', fontsize=fs, transform=ax.transAxes, ha='left', va='bottom', color='white')
                    if j==3:
                        ax.text(0.05, 0.05, 'Residual\nZoom', fontsize=fs, transform=ax.transAxes, ha='left', va='bottom', color='white')

        plt.subplots_adjust(hspace=0.02, wspace=0.02)
        fig.set_size_inches(len(images),4, forward=True)


    def plot_trends(self, wavs, samples, prior_handler, mimical_prior):
        """ For a multi-band fit, plot the 2D model parameter relationship with filter wavelength. """

        # Pull the model parameter keys
        mimical_keys = list(mimical_prior.keys())

        # Convert multi-band samples into model parameter samples
        samples_mimical = np.apply_along_axis(lambda samp: prior_handler.revert(samp).flatten(), 1, samples).reshape(samples.shape[0], len(wavs), len(mimical_keys))

        # Get model parameter posterior stats
        stats = np.percentile(samples_mimical, axis=0, q=(0.15,2.5,16,50,84,97.5,99.85))

        # Loop over model parameters
        fig, ax = plt.subplots(len(mimical_keys)//3+len(mimical_keys)%3, 3)
        for i in range(len(ax.flatten())):

            if i >= len(mimical_keys):
                ax.flatten()[i].set_axis_off()

            else:
                ax.flatten()[i].fill_between(wavs, stats[0].T[i], stats[6].T[i], color='black', alpha=0.05, lw=0)
                ax.flatten()[i].fill_between(wavs, stats[1].T[i], stats[5].T[i], color='black', alpha=0.1, lw=0)
                ax.flatten()[i].fill_between(wavs, stats[2].T[i], stats[4].T[i], color='black', alpha=0.25, lw=0)
                ax.flatten()[i].plot(wavs, stats[3].T[i], color='black')
                ax.flatten()[i].set_ylabel(mimical_keys[i])
                ax.flatten()[i].set_xlabel('$\lambda$')

                '''
                str = ''
                if mimical_prior[mimical_keys[i]][1] == 'Individual':
                    str+=mimical_prior[mimical_keys[i]][1]
                else:
                    str+=f'{mimical_prior[mimical_keys[i]][1]}, {mimical_prior[mimical_keys[i]][2]}'
                if type(mimical_prior[mimical_keys[i]][0]).__name__ == 'tuple':
                    str+='\nFitted'
                else:
                    str+='\nFixed'
                ax.flatten()[i].text(0.95, 0.95, str, fontsize=fs, transform=ax.flatten()[i].transAxes, ha='right', va='top', color='black', bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
                '''

            ax.flatten()[i].xaxis.set_major_locator(AutoLocator())
            ax.flatten()[i].xaxis.set_minor_locator(AutoMinorLocator())
            ax.flatten()[i].yaxis.set_major_locator(AutoLocator())
            ax.flatten()[i].yaxis.set_minor_locator(AutoMinorLocator())
            ax.flatten()[i].tick_params(which='both', direction='in', top=True, right=True)

        fig.set_size_inches(7.03058, ((len(mimical_keys)//3+len(mimical_keys)%3)/3) * 7.03058)
        fig.tight_layout()


    def plot_errors(self, images, wavs, mimical_prior, convolved_models, best_sample, prior_handler, filter_names, segmaps):
        """ Provides a summary of the individual errors used in the analysis based on the maximum likelihood model. """

        # Initiate plot
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=4, ncols=len(images)+1, width_ratios=np.append(np.ones(len(images)), 0.25))

        # Get median Nautilus parameters and transalte into median model parameters.
        param_dict = best_sample
        reverted = prior_handler.revert(param_dict)
        pars = reverted[:,:prior_handler.nmodel]
        rmsarr = reverted[:,prior_handler.nmodel]
        cpfarr = reverted[:,prior_handler.nmodel+1]

       # If user provides RMS values, override prior sample - necessary to recover full arrays
        if not (type(mimical_prior['rms'][0]).__name__ == 'tuple'):
            if (type(mimical_prior['rms'][0]).__name__ == 'list'):
                if (type(mimical_prior['rms'][0][0]).__name__ == 'ndarray'):
                    rmsarr = mimical_prior['rms'][0]
            elif (len(wavs) == 1) & ((type(mimical_prior['rms'][0]).__name__ == 'ndarray')):
                rmsarr = [mimical_prior['rms'][0]]

        # If user provides counts-per-flux parameters, override prior sample - necessary to recover full arrays
        if not (type(mimical_prior['counts_per_flux'][0]).__name__ == 'tuple'):
            if (type(mimical_prior['counts_per_flux'][0]).__name__ == 'list'):
                if ((type(mimical_prior['counts_per_flux'][0][0]).__name__ == 'ndarray')):
                    cpfarr = mimical_prior['counts_per_flux'][0]
            elif (len(wavs) == 1) & (type(mimical_prior['counts_per_flux'][0]).__name__ == 'ndarray'):
                cpfarr = [mimical_prior['counts_per_flux'][0]]

        # Create master lists for appending
        rmserr = []
        poissonerr = []
        sigmaerr = []
        ratio = []

        # Loop over filters to calculate errors
        for i in range(len(wavs)):
            convolved_models[i].parameters = pars[i]
            model = discretize_model(model=convolved_models[i], 
                                    x_range=[0,images[i].shape[1]], 
                                    y_range=[0,images[i].shape[0]], 
                                    mode='center')

            rmsi = (np.zeros_like(images[i]) + rmsarr[i])
            poissonerri = ((cpfarr[i]**(-1/2))*np.sqrt(np.abs(model)))
            sigmaerri = np.sqrt(rmsi**2 + (poissonerri)**2)

            rmserr.append(rmsi)
            poissonerr.append(poissonerri)
            sigmaerr.append(sigmaerri)
            ratio.append(poissonerri/rmsi)

        # Set vmins
        vmins = [-np.max(([x.max() for x in sigmaerr])), 
                 -np.max(([x.max() for x in sigmaerr])), 
                 -np.max(([x.max() for x in sigmaerr])), 
                 0]
        
        # Set vmaxs
        vmaxs = [np.max(([x.max() for x in sigmaerr])), 
                 np.max(([x.max() for x in sigmaerr])), 
                 np.max(([x.max() for x in sigmaerr])), 
                 np.max(([x.max() for x in ratio]))]

        # Initiate colorbars
        ax = fig.add_subplot(gs[0, 0])
        ax.set_axis_off()
        im1 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[0], vmin=vmins[0], cmap=cmc.managua_r, rasterized=True)
        cbarax1 = fig.add_subplot(gs[:3, -1])
        cbarax1.set_yticks([])
        cbarax1.set_xticks([])
        cbar1 = plt.colorbar(im1, cax=cbarax1, fraction=1)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar1.locator = tick_locator
        cbar1.update_ticks()
        im2 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[-1], vmin=vmins[-1], cmap=cmc.managua_r, rasterized=True)
        cbarax2 = fig.add_subplot(gs[3, -1])
        cbarax2.set_yticks([])
        cbarax2.set_xticks([])
        cbar2 = plt.colorbar(im2, cax=cbarax2, fraction=1)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar2.locator = tick_locator
        cbar2.update_ticks()
        
        # Loop over filters and plot
        for i in range(len(wavs)):

            plotims = [rmserr[i], poissonerr[i], sigmaerr[i], ratio[i]]

            for j in range(4):

                ax = fig.add_subplot(gs[j, i])
                im = ax.pcolormesh(plotims[j]* segmaps[i], vmax=vmaxs[j], vmin=vmins[j], cmap=cmc.managua_r, rasterized=True)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_axis_off()

                if j==0:
                    ax.text(0.95, 0.95, filter_names[i].upper(), fontsize=fs, transform=ax.transAxes, ha='right', va='top', color='white')

                if i==0:
                    if j==0:
                        ax.text(0.05, 0.05, 'RMS', fontsize=fs, transform=ax.transAxes, ha='left', va='bottom', color='white')
                    if j==1:
                        ax.text(0.05, 0.05, 'Poisson', fontsize=fs, transform=ax.transAxes, ha='left', va='bottom', color='white')

                    if j==2:
                        ax.text(0.05, 0.05, 'Total Sigma', fontsize=fs, transform=ax.transAxes, ha='left', va='bottom', color='white')

                    if j==3:
                        ax.text(0.05, 0.05, 'Poisson/RMS', fontsize=fs, transform=ax.transAxes, ha='left', va='bottom', color='white')

        plt.subplots_adjust(hspace=0.02, wspace=0.02)
        fig.set_size_inches(len(images),4, forward=True)