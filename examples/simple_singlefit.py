import mimical
import numpy as np
from scipy.stats import uniform, norm
def unif(lower, upper):
    return uniform(loc=lower, scale=upper-lower)

id = '0'

image = [np.load(f'file_to_image.npy')]

psf = [np.load(f'paper_2_sizemass/psf_cecilia/psf_f356w.npy')]

filt_list = ['filters/F200W']


mimical_prior = {}

source_1 = {}
source_1['model'] = mimical.Sersic()
source_1['mag'] = unif(20, 30).isf
source_1['r_eff'] = unif(0.1, 20).isf
source_1['n'] = unif(0.1,10).isf
source_1['x_0'] = unif(83-5, 83+5).isf
source_1['y_0'] = unif(83-5, 83+5).isf
source_1['ellip'] = unif(0, 0.9).isf
source_1['theta'] = unif(0, np.pi).isf
mimical_prior['source_1'] = source_1

mimical_prior['psf_pa'] = unif(-180, 180).isf
mimical_prior['bg'] = norm(0, 1).isf
mimical_prior['rms'] = unif(0, 1).isf
mimical_prior['counts_per_flux'] = 1000000


fit = mimical.fit(str(id), image, filt_list, psf, mimical_prior)

fit.phandler.plot_prior(100, 'rms')
plt.show()


#fit.run(verbose_sampler=True, oversample='auto', pool=12, timeout=60, gpu_acceleration=False, rotmethod='interpolation')
#fit.save_plots()
    