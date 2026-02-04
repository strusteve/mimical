import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution.utils import discretize_model
from tqdm import tqdm
from matplotlib import ticker

class Plotter(object):   

    def plot_median_param(self, images, wavs, convolved_models, samples, fitter_keys, prior_handler, filter_names):

        fig = plt.figure()
        gs = fig.add_gridspec(nrows=4, ncols=images.shape[0]+1, width_ratios=np.append(np.ones(images.shape[0]), 0.25))

        
        # Get median Nautilus parameters and transalte into median model parameters.
        param_dict = dict(zip(fitter_keys, np.median(samples, axis=0)))
        pars = prior_handler.revert(param_dict, wavs)


        models = np.zeros_like(images)
        for i in range(len(wavs)):
            convolved_models[i].parameters = pars[i]
            model = discretize_model(model=convolved_models[i], 
                                    x_range=[0,images[i].shape[1]], 
                                    y_range=[0,images[i].shape[0]], 
                                    mode='center')
            models[i]=model

        
        residuals = images - models

        vmins = [-np.percentile(images.flatten(), q=95), -np.percentile(images.flatten(), q=95), -np.percentile(images.flatten(), q=95), min(np.percentile(residuals.flatten(), q=5), -np.percentile(residuals.flatten(), q=95))]
        vmaxs = [np.percentile(images.flatten(), q=95), np.percentile(images.flatten(), q=95), np.percentile(images.flatten(), q=95), max(-np.percentile(residuals.flatten(), q=5), np.percentile(residuals.flatten(), q=95))]
        cmaps = ['binary', 'binary', 'RdGy']

        ax = fig.add_subplot(gs[0, 0])
        ax.set_axis_off()
        im1 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[0], vmin=vmins[0], cmap='RdGy')
        cbarax1 = fig.add_subplot(gs[:3, -1])
        cbarax1.set_yticks([])
        cbarax1.set_xticks([])
        cbar1 = plt.colorbar(im1, cax=cbarax1, fraction=1)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar1.locator = tick_locator
        cbar1.update_ticks()

        im2 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[-1], vmin=vmins[-1], cmap='RdGy')
        cbarax2 = fig.add_subplot(gs[3, -1])
        cbarax2.set_yticks([])
        cbarax2.set_xticks([])
        cbar2 = plt.colorbar(im2, cax=cbarax2, fraction=1)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar2.locator = tick_locator
        cbar2.update_ticks()

        for i in range(len(wavs)):

            plotims = [images[i], models[i], residuals[i], residuals[i]]

            for j in range(4):

                ax = fig.add_subplot(gs[j, i])
                im = ax.pcolormesh(plotims[j], vmax=vmaxs[j], vmin=vmins[j], cmap='RdGy')
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
        fig.set_size_inches(images.shape[0],4, forward=True)


    def plot_median(self, images, wavs, convolved_models, samples, fitter_keys, prior_handler, filter_names):

        fig = plt.figure()
        gs = fig.add_gridspec(nrows=4, ncols=images.shape[0]+1, width_ratios=np.append(np.ones(images.shape[0]), 0.25))
        
        models = np.zeros((samples.shape[0], *images.shape))

        print("Computing median model image...")
        for j in tqdm(range(samples.shape[0])):
            # Get median Nautilus parameters and transalte into median model parameters.
            param_dict = dict(zip(fitter_keys, samples[j]))
            pars = prior_handler.revert(param_dict, wavs)

            for k in range(len(wavs)):
                convolved_models[k].parameters = pars[k]
                model = discretize_model(model=convolved_models[k], 
                                        x_range=[0,images[k].shape[1]], 
                                        y_range=[0,images[k].shape[0]], 
                                        mode='center')
                models[j,k] = model


        models = np.median(models, axis=0)
        residuals = images - models


        vmins = [-np.percentile(images.flatten(), q=95), -np.percentile(images.flatten(), q=95), -np.percentile(images.flatten(), q=95), min(np.percentile(residuals.flatten(), q=5), -np.percentile(residuals.flatten(), q=95))]
        vmaxs = [np.percentile(images.flatten(), q=95), np.percentile(images.flatten(), q=95), np.percentile(images.flatten(), q=95), max(-np.percentile(residuals.flatten(), q=5), np.percentile(residuals.flatten(), q=95))]
        cmaps = ['binary', 'binary', 'RdGy']

        ax = fig.add_subplot(gs[0, 0])
        ax.set_axis_off()
        im1 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[0], vmin=vmins[0], cmap='RdGy')
        cbarax1 = fig.add_subplot(gs[:3, -1])
        cbarax1.set_yticks([])
        cbarax1.set_xticks([])
        cbar1 = plt.colorbar(im1, cax=cbarax1, fraction=1)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar1.locator = tick_locator
        cbar1.update_ticks()

        im2 = ax.pcolormesh(np.zeros_like(images[0]), vmax=vmaxs[-1], vmin=vmins[-1], cmap='RdGy')
        cbarax2 = fig.add_subplot(gs[3, -1])
        cbarax2.set_yticks([])
        cbarax2.set_xticks([])
        cbar2 = plt.colorbar(im2, cax=cbarax2, fraction=1)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar2.locator = tick_locator
        cbar2.update_ticks()

        for i in range(len(wavs)):

            plotims = [images[i], models[i], residuals[i], residuals[i]]

            for j in range(4):

                ax = fig.add_subplot(gs[j, i])
                im = ax.pcolormesh(plotims[j], vmax=vmaxs[j], vmin=vmins[j], cmap='RdGy')
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
        fig.set_size_inches(images.shape[0],4, forward=True)

