#! /usr/bin/env python
#
#  Copyright 2018 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ISOFIT: Imaging Spectrometer Optimal FITting
# Authors: David R Thompson, david.r.thompson@jpl.nasa.gov
#          Philip G Brodrick, philip.brodrick@jpl.nasa.gov
#          Adam Erickson, adam.m.erickson@nasa.gov
#

import os
import logging
import time

import multiprocessing
import numpy as np
import ray

from isofit.core.forward import ForwardModel
from isofit.inversion.inverse import Inversion
from isofit.inversion.inverse_mcmc import MCMCInversion
from isofit.core.fileio import IO
from isofit.configs import configs


class Isofit:
    """Initialize the Isofit class.

    Args:
        config_file: isofit configuration file in JSON or YAML format
        level: logging level (ERROR, WARNING, INFO, DEBUG)
        logfile: file to write output logs to
    """

    def __init__(self, config_file, row_column='', level='INFO', logfile=None):

        # Explicitly set the number of threads to be 1, so we more effectively
        # run in parallel
        os.environ["MKL_NUM_THREADS"] = "1"

        # Set logging level
        self.loglevel = level
        self.logfile = logfile
        logging.basicConfig(format='%(levelname)s:%(message)s', level=self.loglevel, filename=self.logfile)
        
        logging.info(f'********** BEGINNING LOG FOR RUN ON {time.asctime(time.gmtime())} UTC **********')
        logging.info(f'Reading from config file at: {config_file}')

        self.rows = None
        self.cols = None
        self.config = None

        # Load configuration file
        self.config = configs.create_new_config(config_file)
        self.config.get_config_errors()

        # Initialize ray for parallel execution
        rayargs = {'address': self.config.implementation.ip_head,
                   '_redis_password': self.config.implementation.redis_password,
                   'ignore_reinit_error': self.config.implementation.ray_ignore_reinit_error,
                   '_temp_dir': self.config.implementation.ray_temp_dir,
                    'local_mode': self.config.implementation.n_cores == 1}

        # We can only set the num_cpus if running on a single-node
        if self.config.implementation.ip_head is None and self.config.implementation.redis_password is None:
            rayargs['num_cpus'] = self.config.implementation.n_cores

        if self.config.implementation.debug_mode is False:
            ray.init(**rayargs)
        else:
            logging.info('Debug mode active')

        self.workers = None

    def __del__(self):
        ray.shutdown()

    def run(self, row_column = None):
        """
        Iterate over spectra, reading and writing through the IO
        object to handle formatting, buffering, and deferred write-to-file.
        Attempts to avoid reading the entire file into memory, or hitting
        the physical disk too often.

        row_column: The user can specify
            * a single number, in which case it is interpreted as a row
            * a comma-separated pair, in which case it is interpreted as a
              row/column tuple (i.e. a single spectrum)
            * a comma-separated triplet, in which case it is interpreted as
              a row range in the order (line_start, line_end, 0), for all columns 
              all values are inclusive.
            * a comma-separated quartet, in which case it is interpreted as
              a row, column range in the order (line_start, line_end, sample_start,
              sample_end) all values are inclusive.

            If none of the above, the whole cube will be analyzed.
        """
        
        logging.info("Building first forward model, will generate any necessary LUTs")
        fm = ForwardModel(self.config)
        io = IO(self.config, fm) # Need the n_rows, n_cols for row_column as well
        if row_column is not None:
            ranges = row_column.split(',')
            if len(ranges) == 1:
                # Single row, all columns
                self.rows, self.cols = [int(ranges[0])], range(io.n_cols)
            if len(ranges) == 2:
                # Single row, col (one spectrum)
                self.rows, self.cols = [int(ranges[0])], [int(ranges[1])]
            if len(ranges) == 3:
                # Row range, all columns
                row_start, row_end, col_blank = ranges
                self.rows, self.cols = range(
                    int(row_start), int(row_end)), range(io.n_cols)
            elif len(ranges) == 4:
                # Row, column range
                row_start, row_end, col_start, col_end = ranges
                self.rows = range(int(row_start), int(row_end) + 1)
                self.cols = range(int(col_start), int(col_end) + 1)
        else:
            #io = IO(self.config, fm)
            self.rows = range(io.n_rows)
            self.cols = range(io.n_cols)
        del io # Delete when we're done

        index_pairs = np.vstack([x.flatten(order='f') for x in np.meshgrid(self.rows, self.cols)]).T
        n_iter = index_pairs.shape[0]

        if self.config.implementation.n_cores is None:
            n_workers = multiprocessing.cpu_count()
        else:
            n_workers = self.config.implementation.n_cores

        # Max out the number of workers based on the number of tasks
        n_workers = min(n_workers, n_iter)

        if not self.config.implementation.debug_mode:
            fm_id = ray.put(fm)

        if self.workers is None and not self.config.implementation.debug_mode:
            remote_worker = ray.remote(Worker)
            self.workers = ray.util.ActorPool([remote_worker.remote(self.config, fm_id, self.loglevel, self.logfile, n, n_workers)
                                               for n in range(n_workers)])
        elif self.config.implementation.debug_mode:
            self.workers = Worker(self.config, fm, self.loglevel, self.logfile, 0, 1)

        start_time = time.time()
        n_tasks = min(n_workers * self.config.implementation.task_inflation_factor, n_iter)

        logging.info(f'Beginning {n_iter} inversions in {n_tasks} chunks using {n_workers} cores')

        if not self.config.implementation.debug_mode:
            # Divide up spectra to run into chunks
            index_sets = np.linspace(0, n_iter, num=n_tasks, dtype=int)
            if len(index_sets) == 1:
                indices_to_run = [index_pairs[0:1,:]]
            else:
                indices_to_run = [index_pairs[index_sets[l]:index_sets[l + 1], :]
                                  for l in range(len(index_sets) - 1)]

            res = list(self.workers.map_unordered(lambda a, b: a.run_set_of_spectra.remote(b),
                                              indices_to_run))
        else:
            self.workers.run_set_of_spectra(index_pairs)

        total_time = time.time() - start_time

        logging.info(f'Inversions complete.  {round(total_time,2)}s total, {round(n_iter/total_time,4)} spectra/s, '
                     f'{round(n_iter/total_time/n_workers,4)} spectra/s/core')


class Worker(object):
    def __init__(self, config: configs.Config, forward_model: ForwardModel, loglevel: str, logfile: str, worker_id: int = None, total_workers: int = None):
        """
        Worker class to help run a subset of spectra.

        Args:
            config: isofit configuration
            loglevel: output logging level
            logfile: output logging file
            worker_id: worker ID for logging reference
            total_workers: the total number of workers running, for logging reference
        """

        logging.basicConfig(format='%(levelname)s:%(message)s', level=loglevel, filename=logfile)
        self.config = config
        self.fm = forward_model
        #self.fm = ForwardModel(self.config)

        if self.config.implementation.mode == 'mcmc_inversion':
            self.iv = MCMCInversion(self.config, self.fm)
        elif self.config.implementation.mode in ['inversion', 'simulation']:
            self.iv = Inversion(self.config, self.fm)
        else:
            # This should never be reached due to configuration checking
            raise AttributeError('Config implementation mode node valid')

        self.io = IO(self.config, self.fm)

        self.approximate_total_spectra = None
        if total_workers is not None:
            self.approximate_total_spectra = self.io.n_cols * self.io.n_rows / total_workers
        self.worker_id = worker_id
        self.completed_spectra = 0


    def run_set_of_spectra(self, indices: np.array):


        for index in range(0, indices.shape[0]):

            logging.debug("Read chunk of spectra")
            row, col = indices[index,0], indices[index,1]

            input_data = self.io.get_components_at_index(row, col)

            self.completed_spectra += 1
            if input_data is not None:
                logging.debug("Run model")
                # The inversion returns a list of states, which are
                # intepreted either as samples from the posterior (MCMC case)
                # or as a gradient descent trajectory (standard case). For
                # a trajectory, the last spectrum is the converged solution.
                states = self.iv.invert(input_data.meas, input_data.geom)

                logging.debug("Write chunk of spectra")
                # Write the spectra to disk
                try:
                    self.io.write_spectrum(row, col, states, self.fm, self.iv)
                except ValueError as err:
                    logging.info(
                        f"""
                        Encountered the following ValueError in (row,col) ({row},{col}).
                        Results for this pixel will be all zeros.
                        """
                    )
                    logging.error(err)

                if index % 100 == 0:
                    if self.worker_id is not None and self.approximate_total_spectra is not None:
                        percent = np.round(self.completed_spectra / self.approximate_total_spectra * 100,2)
                        logging.info(f'Worker {self.worker_id} completed {self.completed_spectra}/~{self.approximate_total_spectra}:: {percent}% complete')
                    else:
                        logging.info(f'Worker at start location ({row},{col}) completed {index}/{indices.shape[0]}')

        self.io.flush_buffers()



