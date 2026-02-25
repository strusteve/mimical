import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution.utils import discretize_model
from tqdm import tqdm
from matplotlib import ticker


class plotter(object): 
    """ Contains functions for plotting either a median 
    posterior model (plot_median) or a model based of the median of each
    individual parameter (plot_median_param).
    """


    def plot_median(self, images, wavs, convolved_models, samples, prior_handler, filter_names, segmaps):
        """ Plots the median posterior model."""
        
        # Pass segmaps through images
        for i in range(len(wavs)):
            images[i] *= segmaps[i]

        # Initiate plot
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=4, ncols=len(images)+1, width_ratios=np.append(np.ones(len(images)), 0.25))

        # Create master lists for appending
        master_models = []
        master_residuals = []

        # Loop over samples and wavelengths for generating models
        print("Computing median model image...")
        for j in tqdm(range(samples.shape[0])):
            param_dict = samples[j]
            pars = prior_handler.revert(param_dict)[:,:prior_handler.nmodel]
            models = []
            residuals = []
            for i in range(len(wavs)):
                convolved_models[i].parameters = pars[i]
                model = discretize_model(model=convolved_models[i], 
                                        x_range=[0,images[i].shape[1]], 
                                        y_range=[0,images[i].shape[0]], 
                                        mode='center')
                models.append(model * segmaps[i])
                residuals.append(images[i] - (model * segmaps[i]))
            master_models.append(models)
            master_residuals.append(residuals)

        # Parse model list into median model
        median_models = []
        median_residuals = []
        for i in range(len(wavs)):
            temp_model_arr = np.zeros((samples.shape[0], images[i].shape[0], images[i].shape[1]))
            temp_residuals_arr = np.zeros((samples.shape[0], images[i].shape[0], images[i].shape[1]))
            for j in range(samples.shape[0]):
                temp_model_arr[j] = master_models[j][i]
                temp_residuals_arr[j] = master_residuals[j][i]
            median_models.append(np.median(temp_model_arr, axis=0))
            median_residuals.append(np.median(temp_residuals_arr, axis=0))

        # Set vmins
        vmins = [-max([np.percentile(x.flatten(), q=99) for x in images]), 
                 -max([np.percentile(x.flatten(), q=99) for x in images]), 
                 -max([np.percentile(x.flatten(), q=99) for x in images]), 
                 min( min([np.percentile(x.flatten(), q=1) for x in median_residuals]), -max([-np.percentile(x.flatten(), q=99) for x in median_residuals]))]

        # Set vmaxs
        vmaxs = [max([np.percentile(x.flatten(), q=99) for x in images]), 
                 max([np.percentile(x.flatten(), q=99) for x in images]), 
                 max([np.percentile(x.flatten(), q=99) for x in images]), 
                 max( -min([np.percentile(x.flatten(), q=1) for x in median_residuals]), max([-np.percentile(x.flatten(), q=99) for x in median_residuals]))]

        # Initiate colorbars
        ax = fig.add_subplot(gs[0, 0])
        ax.set_axis_off()
        im1 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[0], vmin=vmins[0], cmap='RdGy', rasterized=True)
        cbarax1 = fig.add_subplot(gs[:3, -1])
        cbarax1.set_yticks([])
        cbarax1.set_xticks([])
        cbar1 = plt.colorbar(im1, cax=cbarax1, fraction=1)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar1.locator = tick_locator
        cbar1.update_ticks()
        im2 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[-1], vmin=vmins[-1], cmap='RdGy', rasterized=True)
        cbarax2 = fig.add_subplot(gs[3, -1])
        cbarax2.set_yticks([])
        cbarax2.set_xticks([])
        cbar2 = plt.colorbar(im2, cax=cbarax2, fraction=1)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar2.locator = tick_locator
        cbar2.update_ticks()
        
        # Loop over filters and plot
        for i in range(len(wavs)):

            plotims = [images[i], median_models[i], median_residuals[i], median_residuals[i]]

            for j in range(4):

                ax = fig.add_subplot(gs[j, i])
                im = ax.pcolormesh(plotims[j], vmax=vmaxs[j], vmin=vmins[j], cmap='RdGy', rasterized=True)
                ax.set_yticks([])
                ax.set_xticks([])

                if j==0:
                    ax.set_title(filter_names[i].upper())

                if i==0:
                    if j==0:
                        ax.set_ylabel('Data')
                    if j==1:
                        ax.set_ylabel('Median\nModel')
                    if j==2:
                        ax.set_ylabel('Residual')
                    if j==3:
                        ax.set_ylabel('Residual\nZoom')
        
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        fig.set_size_inches(len(images),4, forward=True)




    def plot_median_param(self, images, wavs, convolved_models, samples, prior_handler, filter_names, segmaps):
        """ Plots the median parameter posterior model."""

        # Pass segmaps through images
        for i in range(len(wavs)):
            images[i] *= segmaps[i]

        # Initiate plot
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=4, ncols=len(images)+1, width_ratios=np.append(np.ones(len(images)), 0.25))
        
        # Get median Nautilus parameters and transalte into median model parameters.
        param_dict = np.median(samples, axis=0)
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

        # Initiate colorbars
        ax = fig.add_subplot(gs[0, 0])
        ax.set_axis_off()
        im1 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[0], vmin=vmins[0], cmap='RdGy', rasterized=True)
        cbarax1 = fig.add_subplot(gs[:3, -1])
        cbarax1.set_yticks([])
        cbarax1.set_xticks([])
        cbar1 = plt.colorbar(im1, cax=cbarax1, fraction=1)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar1.locator = tick_locator
        cbar1.update_ticks()
        im2 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[-1], vmin=vmins[-1], cmap='RdGy', rasterized=True)
        cbarax2 = fig.add_subplot(gs[3, -1])
        cbarax2.set_yticks([])
        cbarax2.set_xticks([])
        cbar2 = plt.colorbar(im2, cax=cbarax2, fraction=1)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar2.locator = tick_locator
        cbar2.update_ticks()

        # Loop over filters and plot
        for i in range(len(wavs)):

            plotims = [images[i], models[i], residuals[i], residuals[i]]

            for j in range(4):

                ax = fig.add_subplot(gs[j, i])
                im = ax.pcolormesh(plotims[j], vmax=vmaxs[j], vmin=vmins[j], cmap='RdGy', rasterized=True)
                ax.set_yticks([])
                ax.set_xticks([])

                if j==0:
                    ax.set_title(filter_names[i].upper())

                if i==0:
                    if j==0:
                        ax.set_ylabel('Data')
                    if j==1:
                        ax.set_ylabel('Best\nModel')
                    if j==2:
                        ax.set_ylabel('Residual')
                    if j==3:
                        ax.set_ylabel('Residual\nZoom')
        
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        fig.set_size_inches(len(images),4, forward=True)

