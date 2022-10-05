# Run ISOFIT with a specified input state vector that is fixed and does not change
# Useful for testing purposes
# Regina Eckert, October 2022, reckert@jpl.nasa.gov

import numpy as np
from matplotlib import pyplot as plt
from spectral.io import envi
import glob
import os
import json
import pdb

from scipy.spatial.distance import cdist

from skimage.segmentation import slic, mark_boundaries
from utils import dim_red, vis, cos_sim, zero_one
from numpy.linalg import inv
from isofit.utils import apply_oe, extractions
from isofit.core.isofit import Isofit

from numpy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
import scipy
import argparse
import logging
import time

from sklearn.decomposition import PCA

from sklearn.gaussian_process.kernels import RBF

def main(rawargs=None):
    
    parser = argparse.ArgumentParser(description="Apply OE to a block of data.")
    parser.add_argument('input_radiance', type=str)
    parser.add_argument('input_loc', type=str)
    parser.add_argument('input_obs', type=str)
    parser.add_argument('working_directory', type=str)
    parser.add_argument('sensor', type=str)
    parser.add_argument('input_state', type=str)
    parser.add_argument('--copy_input_files', type=int, choices=[0,1], default=0)
    parser.add_argument('--modtran_path', type=str)
    parser.add_argument('--wavelength_path', type=str)
    parser.add_argument('--surface_category', type=str, default="multicomponent_surface")
    parser.add_argument('--aerosol_climatology_path', type=str, default=None)
    parser.add_argument('--rdn_factors_path', type=str)
    parser.add_argument('--surface_path', type=str)
    parser.add_argument('--atmosphere_type', type=str, default='ATM_MIDLAT_SUMMER')
    parser.add_argument('--channelized_uncertainty_path', type=str)
    parser.add_argument('--model_discrepancy_path', type=str)
    parser.add_argument('--lut_config_file', type=str)
    parser.add_argument('--multiple_restarts', type=int, default=0)
    parser.add_argument('--logging_level', type=str, default="INFO")
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--n_cores', type=int, default=1)
    parser.add_argument('--presolve', choices=[0,1], type=int, default=0)
    parser.add_argument('--empirical_line', choices=[0,1], type=int, default=0)
    parser.add_argument('--ray_temp_dir', type=str, default='/tmp/ray')
    parser.add_argument('--emulator_base', type=str, default=None)
    parser.add_argument('--segmentation_size', type=int, default=40)
    parser.add_argument('--err_matrix', type=str, default='ce', choices=['ce','noerr','cosdis','cosdis_norm','pca_cosdis','pca_cosdis_norm'])
    parser.add_argument('--flag_run_part1_only', action='store_true')
    
    parser.add_argument('--l_min',type=float,default=1e-4)
    parser.add_argument('--l_max',type=float,default=1e-1)
    parser.add_argument('--sigf_min',type=float,default=1e-2)
    parser.add_argument('--sigf_max',type=float,default=1e2)
    
    args = parser.parse_args(rawargs)
    
    start_time = time.time()
    ckpt_time = [start_time]
    ckpt_tag  = ['start_all']

    logging.basicConfig(format='%(levelname)s:%(message)s', level=args.logging_level, filename=args.log_file)
    logging.info(f'********** BEGINNING FIXED STATE ISOFIT AT {time.asctime(time.gmtime(start_time))} UTC **********')
    verbose = 0

    # Run the base setup
    logging.info('FI info: Running traditional empirical line version')
    
    args.empirical_line = 1
    
    # Set all the arguments up to be passed to apply_oe
    orig_passargs = args.__dict__
    passargs = []
    for key, item in orig_passargs.items():
        if key not in ['input_radiance', 'input_loc', 'input_obs', 'working_directory', 'sensor']:
            continue
        passargs.append(str(item))
    passargs.extend(['--empirical_line', '1', '--presolve', str(args.presolve), '--surface_path', args.surface_path, '--emulator_base', args.emulator_base, '--n_cores', '40','--log_file', args.log_file, '--wavelength_path', args.wavelength_path, '--segmentation_size', str(args.segmentation_size)])
    
    if args.lut_config_file is not None:
        passargs.extend([ '--lut_config_file', args.lut_config_file])

    logging.basicConfig(format='%(levelname)s:%(message)s', level=args.logging_level, filename=args.log_file)
    
    #***Set up the recon params with apply_oe, but there's got to be a better way to do this
    logging.info('FI info: Passing args to apply_oe')
    ckpt_time.append(time.time())
    ckpt_tag.append('start_applyoe')
    logging.info(f'FI checkpoint: {len(ckpt_time)-1} ({ckpt_tag[-1]}), sub time elapsed: {ckpt_time[-1]-ckpt_time[-2]:.4f} s, total time elapsed: {ckpt_time[-1]-ckpt_time[0]:.4f} s')
    
    #Apply optimal estimation on segmented data with high uncertainty on AOD and water vapor
    apply_oe.main(passargs)
    
    logging.info('FI info: Completed apply_oe')
    ckpt_time.append(time.time())
    ckpt_tag.append('end_applyoe')
    logging.info(f'FI checkpoint: {len(ckpt_time)-1} ({ckpt_tag[-1]}), sub time elapsed: {ckpt_time[-1]-ckpt_time[-2]:.4f} s, total time elapsed: {ckpt_time[-1]-ckpt_time[0]:.4f} s')
    
    logging.info('FI info: Opening files')
    fid = os.path.basename(args.input_radiance).split('_')[0]
   
    rfl_ds = envi.open(os.path.join(args.working_directory, 'output', f'{fid}_rfl.hdr')) #Full size image
    #state_ds = envi.open(os.path.join(args.working_directory, 'output', f'{fid}_subs_state.hdr')) #Segments (not full size image)
    #state = state_ds.open_memmap(interleave='bip')
    #state_unc = envi.open(os.path.join(args.working_directory, 'output', f'{fid}_subs_uncert.hdr')).open_memmap(interleave='bip')
    #loc_subs = envi.open(os.path.join(args.working_directory, 'input', f'{fid}_subs_loc.hdr')).open_memmap(interleave='bip')
    #rfl_sub = np.squeeze(state[1:,:,:425]) # Reflectance loaded from subs state file (segments)
    
    state_input_ds = envi.open(args.input_state) #Segments (not full size image)
    state_input = state_input_ds.open_memmap(interleave='bip')
    
    #***Save ground truth water vapor, AOD to fixed state file
    meta = state_input_ds.metadata.copy()
    meta['bands'] = 2
    meta['band names'] = ['AOT550', 'H2OSTR'] #[state_input_ds.metadata['band names'][idx] for idx in args.input_idx] # ['AOT550', 'H2OSTR']
    del meta['wavelength']
    del meta['wavelength units']
    
    fixed_state_file = os.path.join(args.working_directory, 'output', f'{fid}_fixed_state') #Where state values (besides rfl) will be stored

    fs_ds = envi.create_image(fixed_state_file + '.hdr', meta, ext='', force=True)
    fixed_state = fs_ds.open_memmap(interleave='bip', writable=True)
    fixed_state[...] = state_input[:,:,-2:].copy() 
    
    logging.info('FI info: Setting up the fixed-state version')
    
    #set up config file
    with open(os.path.join(args.working_directory, 'config', f'{fid}_modtran.json'), 'r') as f:
        config = json.load(f) #Load config previously used to run modtran (?? or something)

    config['input']['fixed_state_file'] = os.path.abspath(fixed_state_file) 
    config['output']['estimated_reflectance_file'] = os.path.abspath(os.path.join(args.working_directory, 'output', f'{fid}_refl_fs'))
    config['output']['estimated_state_file'] = os.path.abspath(os.path.join(args.working_directory, 'output', f'{fid}_state_fs'))
    config['output']['posterior_uncertainty_file'] = os.path.abspath(os.path.join(args.working_directory, 'output', f'{fid}_post_uncert_fs'))
    config['input']['loc_file'] = os.path.abspath(args.input_loc)
    config['input']['obs_file'] = os.path.abspath(args.input_obs)
    config['input']['measured_radiance_file'] = os.path.abspath(args.input_radiance)
    del config['output']['atmospheric_coefficients_file']

    fs_config = os.path.join(args.working_directory, 'config', f'{fid}_modtran_fs.json')
    with open(fs_config, 'w') as fout:
        fout.write(json.dumps(config, cls=apply_oe.SerialEncoder, indent=4, sort_keys=True))
    #Write new config file, basic change being to change appropriate paths and include the fixed_state_file as an input
    
    
    logging.info('FI info: Beginning Isofit retrieval with fixed state')
    ckpt_time.append(time.time())
    ckpt_tag.append('start_fixediso')
    logging.info(f'FI checkpoint: {len(ckpt_time)-1} ({ckpt_tag[-1]}), sub time elapsed: {ckpt_time[-1]-ckpt_time[-2]:.4f} s, total time elapsed: {ckpt_time[-1]-ckpt_time[0]:.4f} s')

    model = Isofit(fs_config, level=args.logging_level, logfile=args.log_file)
    model.run()
    
    del model
    logging.info('FI info: Isofit procedure complete')
    logging.info('FI info: SCOE procedure complete')
    
    ckpt_time.append(time.time())
    ckpt_tag.append('end_fixediso_end_all')
    logging.info(f'FI checkpoint: {len(ckpt_time)-1} ({ckpt_tag[-1]}), sub time elapsed: {ckpt_time[-1]-ckpt_time[-2]:.4f} s, total time elapsed: {ckpt_time[-1]-ckpt_time[0]:.4f} s')

    time_tag = time.strftime('%Y%m%dt%H%M%S',time.gmtime(ckpt_time[0]))
    logging.info(f'FI info: Writing timing data to fixediso_time_checkpoints_{time_tag}.txt')
    with open(os.path.join(args.working_directory,'output',f'fixediso_time_checkpoints_{time_tag}.txt'), 'w') as file:
        file.write(','.join(ckpt_tag))
        file.write('\n')
        file.write(','.join("%.18e" % value for value in ckpt_time))
    logging.info(f'********** ENDING FIXED ISOFIT AT {time.asctime(time.gmtime())} UTC **********')


if __name__ == "__main__":
    main()
