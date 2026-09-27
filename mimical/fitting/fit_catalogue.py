import numpy as np
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    rank = 0
    size = 1

from .fit import fit


class fitCatalogue(object):
    """ Fit a catalogue of singly- or multiply-imaged objects with a 2D model
        via Bayesian inference.

    Parameters
    ----------

    runtag : str
        A name for the mimical catalogue run.

    id_list : list
        An ID list for the fitting run. Only really used for output files.

    load_images : function
        Function taking in 'id' and returning a list of images with slices for
        each filter.

    load_filt_list : function
        Function taking in 'id' and returning a  list of path strings to the
        filter transmission curve files, relative to the current working
        directory.

    load_psfs : function
        Function taking in 'id' and returning a list of PSF images with slices
        for each filter.

    load_mimical_prior : function
        The user specified prior which set out the priors for the model
        parameters and passes information about whether to let these vary for
        each filter or whether they follow a power-law or an order-specified
        polynomial relationship.
    """

    def __init__(self, runtag, id_list, load_images, load_filt_list, load_psfs,
                 load_mimical_prior, **kwargs):

        self.runtag = runtag
        self.id_list = id_list
        self.load_images = load_images
        self.load_filt_list = load_filt_list
        self.load_psfs = load_psfs
        self.load_mimical_prior = load_mimical_prior
        self.kwargs = kwargs
        self.done = np.zeros_like(self.id_list, dtype='int')

    def run(self, **run_kwargs):
        """ Runs the nested sampler to sample models, and processes its output.

        Parameters
        ----------

        n_live : int
            Number of live points in nested sampling algorithm.

        make_plots : bool
            Save key plots.
        """

        if size == 1:
            for id in self.id_list:
                single = fit(id, self.load_images(id),
                             self.load_filt_list(id),
                             self.load_psfs(id),
                             self.load_mimical_prior(id),
                             runtag="/"+self.runtag, **self.kwargs)
                single.run(**run_kwargs)
                single.save_output()
            print(f'All {len(self.id_list)} objects done.')

        else:
            if rank == 0:
                # give out first IDs to fit
                for i in range(1, size):
                    if np.min(self.done) == 0:
                        new_id = self.id_list[np.argmin(self.done)]
                        comm.send(new_id, dest=i)
                        self.done[np.argmin(self.done)] = 1
                    else:
                        comm.send(None, dest=i)

                # If all objects are done end
                if np.min(self.done) == 2:
                    return

                while True:  # Add results to catalogue + distribute new IDs
                    done_id, done_rank = comm.recv(source=MPI.ANY_SOURCE)
                    # mark as done
                    self.done[self.id_list == done_id] = 2

                    # Send new ID to process
                    if np.min(self.done) == 0:
                        new_id = self.id_list[np.argmin(self.done != 0)]
                        self.done[self.id_list == new_id] = 1
                        comm.send(new_id, dest=done_rank)
                    else:
                        comm.send(None, dest=done_rank)

                    # Load old ID into catalogue
                    single = fit(done_id, self.load_images(done_id),
                                 self.load_filt_list(done_id),
                                 self.load_psfs(done_id),
                                 self.load_mimical_prior(done_id),
                                 runtag="/"+self.runtag, **self.kwargs)
                    single.run(**run_kwargs)
                    single.save_output(save_catalogue=True,
                                       save_model=False,
                                       save_plots=False)

                    # if all objects done end
                    if np.min(self.done) == 2:
                        print(f'All {len(self.id_list)} objects done.')
                        return

            else:
                while True:
                    id = comm.recv(source=0)
                    if id is None:
                        return

                    single = fit(id, self.load_images(id),
                                 self.load_filt_list(id),
                                 self.load_psfs(id),
                                 self.load_mimical_prior(id),
                                 runtag="/"+self.runtag, **self.kwargs)
                    single.run(**run_kwargs)
                    single.save_output(save_catalogue=False,
                                       save_model=True,
                                       save_plots=True)
                    comm.send([id, rank], dest=0)
