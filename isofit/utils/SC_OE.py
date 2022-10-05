# This is the main program to run our prototype from beginning to the end
# Steffen Mauceri, 07/21
# Edited by Phil Brodrick Fall 2021
# Edited by Regina Eckert, October 2022, reckert

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

import scipy.io as scio

from sklearn.decomposition import PCA

from sklearn.gaussian_process.kernels import RBF

def stable_inv(C):
    
    D, P = scipy.linalg.eigh(C)
    Ds = np.diag(1/np.sqrt(D))
    L = P@Ds
    Cinv = L@L.T
    
    return Cinv

def nll_fn(X_train, Y_train, neg_half_sqdist, noise, cos_dist):
    """
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given
    noise level.

    Args:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise level of Y_train.

    Returns:
        Minimization objective.
    """

    Y_train = Y_train.ravel()

    def nll_stable(theta, diag_regularizer = 0):
        # Numerically more stable implementation of Eq. (11) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.

        # Since neg_half_sqdist is provided here, then use 0 for first two as they are unused
        K = kernel(0, 0, neg_half_sqdist = neg_half_sqdist, l=theta[0], sigma_f=theta[1], dist_conv=1.)
        Kn = (noise ** 2 + diag_regularizer) * np.eye(len(X_train)) + cos_dist #* theta[2]

        K = K + Kn
        
        L = cholesky(K)

        S1 = solve_triangular(L, Y_train, lower=True)
        # L S1 = Y_train; solves for S1 = L\Y_train (knowing that L is lower triangular)
        S2 = solve_triangular(L.T, S1, lower=False)

        #cost = np.sum(np.log(np.diagonal(L))) + \
        #       0.5 * Y_train.dot(S2) + \
        #       0.5 * len(X_train) * np.log(2 * np.pi)

        a = np.sum(np.log(np.diagonal(L)))
        b = 0.5 * Y_train.dot(S2)
        c = 0.5 * len(X_train) * np.log(2 * np.pi) # This does not depend on theta, so unnecessary
        cost = a + b + c

        return cost



    return nll_stable

def kernel(X1, X2, neg_half_sqdist = None, l=1.0, sigma_f=1.0, dist_conv=1):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).
        neg_half_sqdist: used in place of -0.5 * cdist(X1, X2) so that it can be precomputed

    Returns:
        (m x n) matrix.
    """
    if neg_half_sqdist is None:
        neg_half_sqdist = -0.5*cdist(X1, X2)**2

    #kern = sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)
    kern = sigma_f ** 2 * np.exp(neg_half_sqdist / l ** 2)
    return kern

def posterior(X_s, X_train, Y_train, cos_dist, l=100.0, sigma_f=1.0, sigma_y=1e-8, sigma_cos=1.0, diag_regularizer = 0):
    """
    Computes the suffifient statistics of the posterior distribution
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d).
    """
    k_base = kernel(X_train, X_train, l = l, sigma_f = sigma_f) + (sigma_y**2 + diag_regularizer) * np.eye(len(X_train))
    k_bias = sigma_cos*cos_dist
    #k_bias = np.eye(k_bias.shape[0]) * np.sum(k_bias, axis=1)
    K = k_base + k_bias
    
    K_s = kernel(X_train, X_s, l = l, sigma_f = sigma_f)
    #K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = stable_inv(K)

    # Equation (7)
    #mu_s = K_s.T.dot(K_inv).dot(Y_train)
    print(K_s.shape, K_inv.shape, Y_train.shape)
    mu_s = K_s.T @ K_inv @ Y_train

    # Equation (8)
    #cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s#, k_base, k_bias, K_inv #, cov_s

def posterior_cov(X_s, X_train, Y_train, cos_dist, l=100.0, sigma_f=1.0, sigma_y=1e-8, sigma_cos=1.0, diag_regularizer = 0):
    """
    Computes the suffifient statistics of the posterior distribution
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    k_base = kernel(X_train, X_train, l = l, sigma_f = sigma_f) + (sigma_y**2 + diag_regularizer) * np.eye(len(X_train))
    k_bias = sigma_cos*cos_dist
    #k_bias = np.eye(k_bias.shape[0]) * np.sum(k_bias, axis=1)
    K = k_base + k_bias
    
    K_s = kernel(X_train, X_s, l = l, sigma_f = sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = stable_inv(K)

    # Equation (7)
    #mu_s = K_s.T.dot(K_inv).dot(Y_train)
    print(K_s.shape, K_inv.shape, Y_train.shape)
    mu_s = K_s.T @ K_inv @ Y_train

    # Equation (8)
    cov_s = K_ss - K_s.T @ K_inv @ K_s
    return mu_s, np.diagonal(cov_s)


def main(rawargs=None):
    
    parser = argparse.ArgumentParser(description="Apply OE to a block of data.")
    parser.add_argument('input_radiance', type=str)
    parser.add_argument('input_loc', type=str)
    parser.add_argument('input_obs', type=str)
    parser.add_argument('working_directory', type=str)
    parser.add_argument('sensor', type=str)
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
    parser.add_argument('--lut_config_file', type=str,default=None)
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
    parser.add_argument('--gpr_uncert', type=float, default=-1) #When < 0, will use uncertainty from empirical line instead
    parser.add_argument('--flag_skip_inference', action='store_true')
    parser.add_argument('--flag_calc_cov', action='store_true')
    
    parser.add_argument('--l_min',type=float,default=1e-4)
    parser.add_argument('--l_max',type=float,default=1e-1)
    parser.add_argument('--sigf_min',type=float,default=1e-2)
    parser.add_argument('--sigf_max',type=float,default=1e2)
    
    args = parser.parse_args(rawargs)
    
       
    start_time = time.time()
    ckpt_time = [start_time]
    ckpt_tag  = ['start_all']

    logging.basicConfig(format='%(levelname)s:%(message)s', level=args.logging_level, filename=args.log_file)
    logging.info(f'********** BEGINNING SCOE AT {time.asctime(time.gmtime(start_time))} UTC **********')
    verbose = 0
    
    logging.info(args.lut_config_file)

    # Run the base setup
    logging.info('SCOE info: Running traditional empirical line version')
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
    if args.rdn_factors_path is not None:
        passargs.extend([ '--rdn_factors_path',args.rdn_factors_path])
    if args.flag_skip_inference:
        passargs.extend([ '--flag_skip_inference'])
    if args.multiple_restarts > 0:
        passargs.extend([ '--multiple_restarts'])

    
    logging.info('SCOE info: Passing args to apply_oe')
    ckpt_time.append(time.time())
    ckpt_tag.append('start_applyoe')
    logging.info(f'SCOE checkpoint: {len(ckpt_time)-1} ({ckpt_tag[-1]}), sub time elapsed: {ckpt_time[-1]-ckpt_time[-2]:.4f} s, total time elapsed: {ckpt_time[-1]-ckpt_time[0]:.4f} s')
    
    #Apply optimal estimation on segmented data with high uncertainty on AOD and water vapor
    apply_oe.main(passargs)
    
    logging.info('SCOE info: Completed apply_oe')
    ckpt_time.append(time.time())
    ckpt_tag.append('end_applyoe')
    logging.info(f'SCOE checkpoint: {len(ckpt_time)-1} ({ckpt_tag[-1]}), sub time elapsed: {ckpt_time[-1]-ckpt_time[-2]:.4f} s, total time elapsed: {ckpt_time[-1]-ckpt_time[0]:.4f} s')
    #apply_oe.main(args.input_radiance, args.input_loc, args.input_obs, args.working_directory, args.sensor, **passargs)
    #*** Saves information in : XXX that is then read out later
    #This information is the AOD, and water vapor for each segment (?)
    #Plus, the empirical line estimation of all the reflectances saved in the (...)_rfl file 
    # --> this is not actually used here, but is an artifact of using apply_oe as a way to 
    #     get the segments set up & to estimate the AOD and water vapor for each segment

    logging.info('SCOE info: Opening files in SC_OE')
    fid = os.path.basename(args.input_radiance).split('_')[0]
    
    
    # ## use GPR to adjust OE results for AOD and H2O
    # # calculate cosine similarity
    # rdn_sub = envi.open(os.path.join(args.working_directory, 'input', f'{fid}_subs_rdn.hdr')).load()[1:,:,:] 
    # # First element is dropped because it is all zeroes 
    # # ***(unclear why it is all zeroes - is that always the case or just for this particular data??)
    # rdn_PC_sub = dim_red(rdn_sub, min(10, rdn_sub.shape[0])) #Calculates normalized PCA basis components on spatial domain
    # #Captures the principle components of variation across wavelength on spatial domain (components are spatial)
    # cos_sim_sub = cos_sim(np.squeeze(rdn_PC_sub)) #Calculates cosine similarity (i.e. inner products) between the principal components at each pixel
    # #*** This is actually unused as a value, but could be tried as another way to estimate the error...

    #good_bands = np.loadtxt('good_bands.txt').astype(np.int8)

    #*** Editing out need for the rfl ds!!
    rfl_ds = envi.open(os.path.join(args.working_directory, 'output', f'{fid}_rfl.hdr')) #Full size image
    
    # input_radiance_img = envi.open(args.input_radiance)
    # output_metadata = input_radiance_img.metadata
    # output_metadata['interleave'] = 'bil'
    
    state_ds = envi.open(os.path.join(args.working_directory, 'output', f'{fid}_subs_state.hdr')) #Segments (not full size image)
    state = state_ds.open_memmap(interleave='bip')
    state_unc = envi.open(os.path.join(args.working_directory, 'output', f'{fid}_subs_uncert.hdr')).open_memmap(interleave='bip')
    loc_subs = envi.open(os.path.join(args.working_directory, 'input', f'{fid}_subs_loc.hdr')).open_memmap(interleave='bip')
    rfl_sub = np.squeeze(state[1:,:,:425]) # Reflectance loaded from subs state file (segments)
    #***Again, unclear why first segment is skipped (because it is all zeroes, but why is it all zeroes?)

    fixed_state_file = os.path.join(args.working_directory, 'output', f'{fid}_fixed_state') #Where state values (besides rfl) will be stored
    if args.flag_calc_cov:
        fixed_state_cov_file = os.path.join(args.working_directory, 'output', f'{fid}_fs_cov') #Where state value covariance will be stored
    
    if os.path.isfile(fixed_state_file):
        logging.info("SCOE info: Previous fixed state file found, using this")
        ckpt_time.append(time.time())
        ckpt_tag.append('end_env')
        logging.info(f'SCOE checkpoint: {len(ckpt_time)-1} ({ckpt_tag[-1]}), sub time elapsed: {ckpt_time[-1]-ckpt_time[-2]:.4f} s, total time elapsed: {ckpt_time[-1]-ckpt_time[0]:.4f} s')
    else:
        logging.info('SCOE info: Running the GPR')
        ckpt_time.append(time.time())
        ckpt_tag.append('start_gpr')
        logging.info(f'SCOE checkpoint: {len(ckpt_time)-1} ({ckpt_tag[-1]}), sub time elapsed: {ckpt_time[-1]-ckpt_time[-2]:.4f} s, total time elapsed: {ckpt_time[-1]-ckpt_time[0]:.4f} s')
        
        #v5bounds
        length_scale_b = np.array([[1e-3, 1e-1],[1e-5,1e-3]]) #AOD, water vapor #np.array([[1e-4, 1e-1],[1e-8,1e-5]]) #AOD, water vapor
        sigmaf_b = np.array([[1e-3,1e0],[1e-3,1e2]]) #AOD, water vapor
        # #gpr_uncert = np.array([0.01,0.01]) #AOD, water vapor
        
        #v6bounds
        # length_scale_b = np.array([[1e-4, 1e-1],[1e-7,1e-2]]) #AOD, water vapor
        # sigmaf_b = np.array([[1e-4,1e0],[1e-3,1e3]]) #AOD, water vapor
        
        states_to_run = ['AOT550', 'H2OSTR']
        if args.flag_calc_cov:
            A_cov_combined = np.zeros((rfl_ds.shape[0],rfl_ds.shape[1], len(states_to_run)))
        A_combined = np.zeros((rfl_ds.shape[0],rfl_ds.shape[1], len(states_to_run))) #[Image size (full), # states]
        restarts = 20  # number of times we want to restart the minimzation
        
        l_opt_save = np.zeros(len(states_to_run))
        sigmaf_opt_save = l_opt_save.copy()
        init_opt = np.zeros((restarts,2,len(states_to_run)))
        cost_opt = np.zeros((restarts,len(states_to_run)))
        l_opt_all      = cost_opt.copy()
        sigmaf_opt_all = cost_opt.copy()
        n_iter_opt = cost_opt.copy()
    
        for _s, statename in enumerate(states_to_run):

            # For -1 = water vapor, -2 = AOD.  #todo - grab dynamically from header
            state_ind = state_ds.metadata['band names'].index(statename)
            A_sub = np.squeeze(state[...,state_ind].copy())[1:] #State values from segments
            A_uncert_sub = np.squeeze(state_unc[...,state_ind].copy())[1:] #Uncertainty from that state (also segments)
            X_sub = np.squeeze(loc_subs[...,:-1].copy())[1:,:] #Loc information for that state (segments); 
            # *** except last value (which is???)

            # Load pre-made correlated error models for the different states
            #*** What are these? How are they made?
            if statename == 'AOT550':
                CEM_filename = 'research/data/aod_correlated_error_model.txt'
            elif statename == 'H2OSTR':
                CEM_filename = 'research/data/h2o_correlated_error_model.txt'
            else:
                raise ValueError(f'{statename} is an unsupported state variable')

            if args.err_matrix == 'ce':
                CEM = CorrelatedErrorModel(CEM_filename)
                cos_sim_sub = CEM.error_covariance(rfl_sub, rfl_sub)
            elif args.err_matrix == 'cosdis':
                #cos_sim_sub = cos_sim(rfl_sub)
                cos_sim_sub = cos_sim(rfl_sub-np.mean(rfl_sub,axis=0,keepdims=True))
            elif args.err_matrix == 'cosdis_norm':
                cos_sim_sub = cos_sim((rfl_sub-np.mean(rfl_sub,axis=0,keepdims=True)) / np.std(rfl_sub,axis=0,keepdims=1))
            elif args.err_matrix == 'pca_cosdis':
                #rfl_flat = np.reshape(rfl_sub, (np.shape(rfl_sub)[0] * np.shape(rfl_sub)[1], -1)) #With only shape[1] = 1, this performs np.squeeze()
                rfl_flat = np.squeeze(rfl_sub)
                print(rfl_flat.shape)
                rfl_flat = (rfl_flat - np.mean(rfl_flat, 0)) / np.std(rfl_flat, 0) #subtract spatial mean and normalize by spatial std for each wavelength
                pca = PCA(n_components=10)
                PC = pca.fit_transform(rfl_flat) #Will do PCA of flattened spatial signatures
                PC_meanSub = PC - np.mean(PC,axis=1,keepdims=True) #Mean subtracting & normalizing
                cos_sim_sub = cos_sim(PC_meanSub)
            elif args.err_matrix == 'pca_cosdis_norm':
                #rfl_flat = np.reshape(rfl_sub, (np.shape(rfl_sub)[0] * np.shape(rfl_sub)[1], -1)) #With only shape[1] = 1, this performs np.squeeze()
                rfl_flat = np.squeeze(rfl_sub)
                rfl_flat = (rfl_flat - np.mean(rfl_flat, 0)) / np.std(rfl_flat, 0) #subtract spatial mean and normalize by spatial std for each wavelength
                pca = PCA(n_components=10)
                PC = pca.fit_transform(rfl_flat) #Will do PCA of flattened spatial signatures
                PC_meanSub = PC - np.mean(PC,axis=1,keepdims=True) #Mean subtracting & normalizing
                cos_sim_sub = cos_sim(PC_meanSub/np.std(PC_meanSub,axis=1,keepdims=True))
            elif args.err_matrix == 'noerr':
                # cos_sim_sub = cos_sim(rfl_sub)
                # cos_sim_sub[...] = 0
                cos_sim_sub = np.zeros_like(cos_sim(rfl_sub)) #Terrible way to do this, but more readable

            # Minimize the negative log-likelihood and find hyperparamaters for GPR
            neg_half_sqdist = -0.5*cdist(X_sub, X_sub)**2 # Pre-compute outside of optimization
            #res = minimize(nll_fn(X_sub, A_sub, neg_half_sqdist, A_uncert_sub, cos_sim_sub),
            #               [0.02, 10, 0.1],   #initial values: l, sigma_f, cos_dist scaling
            #               bounds=((1e-5, 1e0), (1e-5, 1e2),(1e-5, 1)), 
            #               method='L-BFGS-B', options={'ftol' :1e-3, 'iprint': 1})

            # Minimize the negative log-likelihood w.r.t. parameters l and sigma_f.

            #Original bounds, through April 1 2022
            #b = np.array([[1e-4, 1e-1], [1e-2, 1e2], [1e-5, 1]])  # length scale, BRF multiplier, factor for correlated noise
            #Test bounds April 2022
            #b = np.array([[1e-8, 1e-1], [1e-6, 1e2], [1e-5, 1]])  # length scale, BRF multiplier, factor for correlated noise
            #b = np.array([[args.l_min, args.l_max],[args.sigf_min,args.sigf_max]])
            b = np.array([[length_scale_b[_s,0], length_scale_b[_s,1]],[sigmaf_b[_s,0], sigmaf_b[_s,1]]])
            b_log10 = np.log10(b)
            
            best = np.inf
            #init = np.random.rand(restarts, len(b)) * (b[:, 1] - b[:, 0]) + b[:, 0]
            init = np.power(10,np.random.rand(restarts, len(b_log10)) * (b_log10[:, 1] - b_log10[:, 0]) + b_log10[:, 0]) #Make sure we're initializing from across orders of magnitude
            init_opt[:,:,_s] = init
            
            if args.gpr_uncert < 0:
                #Use the uncertainty produced by empirical line
                gpr_uncert_sub = A_uncert_sub
                logging.info('SCOE info: Using uncertainty output from empirical line as noise input to GPR')
            else:
                #Otherwise use uncertainty provided
                gpr_uncert_sub = args.gpr_uncert*np.ones(A_uncert_sub.shape)
                logging.info('SCOE info: Using static value {} as noise input to GPR'.format(args.gpr_uncert))
                
            
            for i in range(restarts):
                res = minimize(nll_fn(X_sub, A_sub - np.mean(A_sub), neg_half_sqdist, gpr_uncert_sub, cos_sim_sub),
                               init[i, :2],  # initial values
                               #bounds=((b[0, 0], b[0, 1]), (b[1, 0], b[1, 1]), (b[2, 0], b[2, 1])),
                               bounds=((b[0, 0], b[0, 1]), (b[1, 0], b[1, 1])),
                               method='L-BFGS-B')
                
                cost_opt[i,_s]  = res.fun
                l_opt_all[i,_s] = res.x[0]
                sigmaf_opt_all[i,_s] = res.x[1]
                n_iter_opt[i,_s] = res.nfev
            
                #Could also check convergence status before saving the result
                if res.fun < best:
                    best = res.fun
                    best_res = res
            res = best_res
            
            if not res.success:
                logging.info('SCOE info: Warning: Best result was not converged!\nOptimzer output:')
                logging.info(res)
                
            # Store the optimization results in global variables for prediction with GP
            #l_opt, sigma_f_opt, sigma_cos_opt = res.x
            l_opt, sigma_f_opt = res.x
            sigma_cos_opt = 1
            #l_opt, sigma_f_opt, sigma_cos_opt = 0.02, 0.2, 0.2
            
            l_opt_save[_s] = l_opt
            sigmaf_opt_save[_s] = sigma_f_opt

            X = envi.open(args.input_loc + '.hdr').open_memmap(interleave='bip')[...,:-1].copy()
            # make prediction with GP for each pixel in the scene
            if args.flag_calc_cov:
                logging.info('SCOE info: Calculating posterior mean and covariance')
                A_pred, A_cov = posterior_cov(X.reshape(-1, 2), X_sub, A_sub - np.mean(A_sub), cos_sim_sub,
                l=l_opt, sigma_f=sigma_f_opt, sigma_y=A_uncert_sub,sigma_cos=sigma_cos_opt)
            else:
                logging.info('SCOE info: Calculating posterior mean only')
                A_pred = posterior(X.reshape(-1, 2), X_sub, A_sub - np.mean(A_sub), cos_sim_sub,
                l=l_opt, sigma_f=sigma_f_opt, sigma_y=A_uncert_sub,sigma_cos=sigma_cos_opt)

            A_shape = X.shape[0:2]
            if verbose:
                A_min = np.min(A_sub)
                A_max = np.max(A_sub)
                vis(A_pred, A_shape, fid + '\n smoothed by GPR', A_min, A_max, True)

            A_combined[...,_s] = A_pred.reshape(A_shape) + np.mean(A_sub)
            if args.flag_calc_cov:
                A_cov_combined[...,_s] = A_cov.reshape(A_shape)
            
            np.savetxt(os.path.join(args.working_directory,'output',f'{statename}_gpr_coefficients.txt'), [l_opt, sigma_f_opt])
            logging.info(f'SCOE info: Done with GPR for {statename} with error method {args.err_matrix}')
            ckpt_time.append(time.time())
            ckpt_tag.append(f'end_gpr_{statename}')
            logging.info(f'SCOE checkpoint: {len(ckpt_time)-1} ({ckpt_tag[-1]}), sub time elapsed: {ckpt_time[-1]-ckpt_time[-2]:.4f} s, total time elapsed: {ckpt_time[-1]-ckpt_time[0]:.4f} s')

        logging.info('SCOE info: Writing GPR prediction to disk')
        meta = rfl_ds.metadata.copy()
        meta['bands'] = 2
        meta['band names'] = ['AOT550', 'H2OSTR']
        del meta['wavelength']
        del meta['wavelength units']
        
        fs_ds = envi.create_image(fixed_state_file + '.hdr', meta, ext='', force=True)
        fixed_state = fs_ds.open_memmap(interleave='bip', writable=True)
        fixed_state[...] = A_combined
        del fixed_state
        
        if args.flag_calc_cov:
            fs_cov_ds = envi.create_image(fixed_state_cov_file + '.hdr', meta, ext='', force=True)
            fixed_state_cov = fs_cov_ds.open_memmap(interleave='bip', writable=True)
            fixed_state_cov[...] = A_cov_combined
            del fixed_state_cov
        
        mdict = {'l_opt_all': l_opt_all,
             'sigmaf_opt_all': sigmaf_opt_all,
             'cost_opt': cost_opt,
             'n_iter_opt': n_iter_opt,
             'l_opt_save': l_opt_save,
             'sigmaf_opt_save': sigmaf_opt_save,
             'init_opt': init_opt,
             'length_scale_b': length_scale_b,
             'sigmaf_b': sigmaf_b}
        
        scio.savemat(fixed_state_file + '_gpr.mat',mdict)
        del mdict
       
    logging.info('SCOE info: Finished with GPR processing')
    ckpt_time.append(time.time())
    ckpt_tag.append('end_gpr_all')
    logging.info(f'SCOE checkpoint: {len(ckpt_time)-1} ({ckpt_tag[-1]}), sub time elapsed: {ckpt_time[-1]-ckpt_time[-2]:.4f} s, total time elapsed: {ckpt_time[-1]-ckpt_time[0]:.4f} s')
    
    if args.flag_run_part1_only:
        logging.info('SCOE info: Halting after GPR of atmospheric params')
        quit()
    else:
        logging.info('SCOE info: Continuing to final recovery of reflectance with fixed atmospheric params out of GPR')
            
    logging.info('SCOE info: Setting up the fixed-state version')
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
    
    if config['implementation']['inversion'].get('inversion_grid_as_preseed'):
        #If used multiple restarts flag, avoid errors because now do not want to do mult restarts w/ fixed state
        del config['implementation']['inversion']['inversion_grid_as_preseed']
        del config['implementation']['inversion']['integration_grid']

    fs_config = os.path.join(args.working_directory, 'config', f'{fid}_modtran_fs.json')
    with open(fs_config, 'w') as fout:
        fout.write(json.dumps(config, cls=apply_oe.SerialEncoder, indent=4, sort_keys=True))
    #Write new config file, basic change being to change appropriate paths and include the fixed_state_file as an input
    logging.info('SCOE info: Beginning Isofit retrieval with fixed state')
    ckpt_time.append(time.time())
    ckpt_tag.append('start_fixediso')
    logging.info(f'SCOE checkpoint: {len(ckpt_time)-1} ({ckpt_tag[-1]}), sub time elapsed: {ckpt_time[-1]-ckpt_time[-2]:.4f} s, total time elapsed: {ckpt_time[-1]-ckpt_time[0]:.4f} s')
    model = Isofit(fs_config, level=args.logging_level, logfile=args.log_file)
    model.run()
    del model
    logging.info('SCOE info: Isofit procedure complete')
    logging.info('SCOE info: SCOE procedure complete')
    ckpt_time.append(time.time())
    ckpt_tag.append('end_fixediso_end_all')
    logging.info(f'SCOE checkpoint: {len(ckpt_time)-1} ({ckpt_tag[-1]}), sub time elapsed: {ckpt_time[-1]-ckpt_time[-2]:.4f} s, total time elapsed: {ckpt_time[-1]-ckpt_time[0]:.4f} s')
    
    time_tag = time.strftime('%Y%m%dt%H%M%S',time.gmtime(ckpt_time[0]))
    logging.info(f'SCOE info: Writing timing data to scoe_time_checkpoints_{time_tag}.txt')
    with open(os.path.join(args.working_directory,'output',f'scoe_time_checkpoints_{time_tag}.txt'), 'w') as file:
        file.write(','.join(ckpt_tag))
        file.write('\n')
        file.write(','.join("%.18e" % value for value in ckpt_time))
    logging.info(f'********** ENDING SCOE AT {time.asctime(time.gmtime())} UTC **********')

class CorrelatedErrorModel:
    ''' This model predicts the correlated atmospheric estimation error
      associated with a list of surface reflectance spectra.  The 
      constructor requires a filepath to a model file.  The model file
      has the format of a single header line specifying an RBF width, 
      and a single ASCII column of text specifying a linear operator.'''

    def __init__(self, filepath):
        with open(filepath,'r') as fin:
            header = fin.readline()
            self.kernel_width = float(header.strip().split()[-1])
            self.kernel = RBF(length_scale = self.kernel_width)
            self.linear_model = np.array([float(line) for line in fin.readlines()])

    def error_covariance(self,X,Y):
        if (len(self.linear_model) != X.shape[1]) or \
           (len(self.linear_model) != Y.shape[1]):
            raise IndexError('Mismatched error model dimensions!')

        # Multiply our linear model to all spectra
        Xn = np.array([x.dot(self.linear_model) for x in X])
        Yn = np.array([y.dot(self.linear_model) for y in Y])

        # Calculate the RBF Kernel Matrix
        return self.kernel(Xn[:,np.newaxis],Yn[:,np.newaxis]) 


if __name__ == "__main__":
    main()
