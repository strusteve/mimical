**Mimical** (**M**\odelling the **I**\ntensity of **M**\ultiply-**I**\maged **C**\elesti\ **A**\l **L**\ight)


Mimical is an intensity modelling code optimised for multiply-imaged objects (image cubes), performing simultaneous Bayseian inference of model parameters via the nested sampling algorithm. Mimical currently supports Sersic and Point-source submodels, but users can easily bolt-on any other 2D submodel. Since every new filter introduces new submodel free parameters, Mimical supports the assumption of a user defined polynomial or power-law dependency with image wavelength, automatically sampling their coefficients over the allowed parameter region. Additionally, with support for CPU parallelisation via MPI and GPU acceleration via PyTorch, Mimical has significant computational flexibility, allowing users to tune their settings based on available hardware.


**Installation**

Mimical can be installed with pip:

.. code::

    pip install mimical

**Required input**

#. ``images`` - A list of images with slices for each filter.
#. ``filt_list`` - A list of path strings to the filter transmission curve files.
#. ``psfs`` - A list of PSF images with slices for each filter. (Normalised to 1)
#. ``mimical_prior`` - The Mimical prior (*see below*)

**Mimical prior**

Below is an example ``mimical_prior`` for a run using the default Sersic submodel. The first set of element keys must have ``'source'`` and themselves contain keys that match the submodel parameter names. Users can include any number of these to fit multiple-component models. Following this, the next element, named ``psf_pa``, traces the rotation of the PSF. The final four elements must be named ``bg``, ``rms``, ``counts_per_flux`` and ``correlation_width``. Parameters can be assigned a constant prior between filters by setting a callable, or varied independently between filters by setting a tuple or callables. Parameters can also be assumed to have a polynomial/power-law relationship with wavelength by setting ''``(parameter bounds, 'polynomial'/'power-law', polynomial order / power bounds)``'', e.g ''``((0,10), 'polynomial', 3)``''.

.. code::
     
     source_1 = {}
     source_1['model'] = mimical.Sersic()
     source_1['mag'] = uniform(20, 25)
     source_1['r_eff'] = ((0.1, 20), 'polynomial', 1)
     source_1['n'] = ((0.1, 10), 'power-law', (-3,3))
     source_1['x_0'] = norm(50, 3)
     source_1['y_0'] = norm(50, 3)
     source_1['ellip'] = uniform(0, 0.9)
     source_1['theta'] = uniform(0, np.pi)

     mimical_prior = {}
     mimical_prior['source_1'] = source_1
     mimical_prior['psf_pa'] = uniform(-180, 180)
     mimical_prior['bg'] = 0.
     mimical_prior['rms'] = 'infer'
     mimical_prior['counts_per_flux'] = cpf_list
     mimical_prior['correlation_width'] = 1.

The ``bg`` parameter traces the background level in the image, set to zero if you trust the background subtraction. The ``rms`` parameter traces the sky background RMS noise in the image. The ``counts_per_flux`` parameter allows Mimical to associate poisson uncertainty with the generated model, this can effectively be ignored by setting it to a large number e.g. 1e50. The ``correlation_width`` parameter specifies the kernel-dependent width defining the spatially correlated noise term used in the likelihood covariance. This can be ignored by setting it to 1.0 with the default triangular kernel, as this is equivalent to an independent noise assumption.


**Optional input and parameters**

``fit(id, images, filt_list, psfs, mimical_prior, ...)``

* ``se_clean = False`` : Whether or not to let SourceExtractor clean the input images of contaminants. Must allow 'sex' command via terminal.
* ``se_maxdist = 10`` : The distance after which the closest detected source is considered a contaminant. Necessary for images in which the target is undetected.
* ``dilute = True`` : Whether or not to apply a circular miminum filter over the contamination map to dilute it.
* ``dilute_radius = 3`` : If dilute is ``True``, apply minimim filter with radius ``dilute_radius`` over the contamination map.

``fit.run(...)``

* ``n_live = 1000`` : The nested sampling live points.
* ``pool = None`` : Number of cores to parallelise likelihood calls.
* ``oversample = None`` : Oversampling factor..
* ``oversample_bl = None`` : Central box width in which to oversample if desired.
* ``oversample_radii = None`` : Radii defining annuli in which to oversample (oversample must be a list).
* ``gpu_acceleration = False`` : Whether or not to generate models on the GPU.
* ``verbose_sampler = True`` : Whether or not to show verbose Nautilus output.
* ``timeout = np.inf`` : Timeout for nested sampling.
* ``rotmethod = 'interpolation'`` : Method for rotating PSFs
* ``corrkernel = Kernels.triang`` : Correlation kernel
* ``auto_nref = 10000`` : Reference image oversampling when using automatic oversampling


**Fixing parameters**

You can fix any of the parameters in the Mimical prior by setting the parameter prior to either a float/int (constant across images) or a list (specific to each image). For instance, to keep ``x_0`` constant across all images, one would pass a float/int. Or, to supply the ``r_eff`` for each image separately, one would pass a list of floats. For the parameters ``bg``, ``rms`` and ``counts_per_flux`` can also be supplied as an ndarray or list of ndarrays in the same shape as ``images``, while the corner plot samples will show the mean of these images for generality, the full arrays will be parsed in the likelihood function.


**Parallelisation**

Mimical can be parallelised to different cores in one of two ways:

* The likelihood calculations can be parallelised to different cores by using the ``pool`` keyword argument in the ``run`` function. This is ideal for single object fits.
* When using ``fit_catalogue``, the cript can be run using mpi4py to parallelise fits to individual cores. E.g., ``mpirun/mpiexec -n [ncores] python [filename].py``. This is ideal for large catalogue fits.

Running Mimical with both of these options enabled is **untested** but probably infeasible.

**GPU acceleration**

The most computationally expensive step in Mimical is the model generation, which takes place in each likelihood call when a new set of model parameters is sampled. This step has been vectorised and written with PyTorch tensors, such that if a compatible GPU is available (such as an Nvidia card or apple silicon M-series chip) the model generation can be accelerated.