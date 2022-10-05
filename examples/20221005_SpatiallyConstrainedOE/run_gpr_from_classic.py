# Run GPR on randomly chosen points from atmospheric states
# produced by pixel-by-pixel processing (which is already done and called in here)
# Simulates picking random points to use for input to GPR without re-processing
# Regina Eckert, October 2022, reckert@jpl.nasa.gov
from spectral.io import envi
import numpy as np
import os, sys
import warnings
warnings.filterwarnings('ignore')

import time
from scipy.spatial.distance import cdist

from utils import dim_red, vis, cos_sim, zero_one
from numpy.linalg import inv, cholesky

from scipy.linalg import solve_triangular, eigh
from scipy.optimize import minimize
import scipy.io as scio

from sklearn.decomposition import PCA

from sklearn.gaussian_process.kernels import RBF

def stable_inv(C):
    
    D, P = eigh(C)
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
        Posterior mean vector (n x d) and covariance matrix (n x n).
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
    
#     k_base = kernel(X_train, X_train, l = l, sigma_f = sigma_f) + (sigma_y**2 + diag_regularizer) * np.eye(len(X_train))
#     k_bias = sigma_cos*cos_dist
#     #k_bias = np.eye(k_bias.shape[0]) * np.sum(k_bias, axis=1)
#     K = k_base + k_bias

#     K_s = kernel(X_train, X_s, l = l, sigma_f = sigma_f)
#     K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
#     K_inv = stable_inv(K)

#     # Equation (7)
#     #mu_s = K_s.T.dot(K_inv).dot(Y_train)
#     print(K_s.shape, K_inv.shape, Y_train.shape)
#     mu_s = K_s.T @ K_inv @ Y_train

#     # Equation (8)
#     cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s#, k_base, k_bias, K_inv #, cov_s

def main():
    
    classic_tag_list = ['_noise1.0inst','_noise10.0inst'] #['_noise2.0inst','_noise5.0inst'] #'_noise1.0inst_presolve','_noise10.0inst'
    output_tag_list = ['_noise1.0inst','_noise10.0inst'] #['_noise2.0inst','_noise5.0inst'] #'_noise1.0inst_presolve','_noise10.0inst'
    
    classic_outtag_list = ['_aodinit0.25','_aodinit0.25']
    
    for kk in range(len(classic_tag_list)):
        # Load all
        instrument_tag = 'ang20210710t100946'
        sim_tag = 'simplain_full_raw'
        exp_tag = 'indiawater_gaussaod0.2{}'.format(classic_tag_list[kk])
        classic_outtag = classic_outtag_list[kk]
        retrieval_tag = '_v5b_randomPts'
        retrieval_base_list = ['scoe_retrievals_noerr_pts']#,'scoe_retrievals_ce','scoe_retrievals_cosdis']#,'scoe_retrievals_ce','scoe_retrievals_cosdis','scoe_retrievals_noerr']

        title_base_list = ['scoe, noerr, pts']#,'scoe, ce (corr error)','scoe, cosdis']

        base_dir = '.'
        retrieval_dir_list = [base_dir + 'retrievals/{}/{}_{}_{}{}{}/'.format(retrieval_base_list[ii],instrument_tag,sim_tag,exp_tag,classic_outtag,retrieval_tag) for ii in range(len(retrieval_base_list))] 
        data_dir = base_dir + 'data/{}_{}_{}/'.format(instrument_tag,sim_tag,exp_tag)
        classic_dir = base_dir + 'retrievals/classic_retrievals/{}_{}_{}{}/'.format(instrument_tag,sim_tag,exp_tag,classic_outtag)

        if not os.path.isdir(retrieval_dir_list[0]):
            os.makedirs(retrieval_dir_list[0])

        # Load wavelength and trim bands
        wl = np.genfromtxt(base_dir + 'data/wl.txt')
        good_bands = np.loadtxt(base_dir + 'data/good_bands.txt').astype(np.int8)
        wl_nan = wl.copy()
        wl_nan[np.logical_not(good_bands)] = np.nan

        # Load true state and simulated radiance
        print('Loading {}...'.format(data_dir +'{}_{}_{}_simulated_rdn.hdr'.format(instrument_tag,sim_tag,exp_tag)))
        synth_data = envi.open(data_dir +'{}_{}_{}_simulated_rdn.hdr'.format(instrument_tag,sim_tag,exp_tag)).open_memmap(interleave='bip')
        print('Loading {}...'.format(data_dir+ '{}_{}_{}_true_state.hdr'.format(instrument_tag,sim_tag,exp_tag)))
        true_state = envi.open(data_dir+ '{}_{}_{}_true_state.hdr'.format(instrument_tag,sim_tag,exp_tag)).open_memmap(interleave='bip')[:,:,-2:]

        # Load classic
        print('Loading {}...'.format(classic_dir + 'output/{}_state.hdr'.format(instrument_tag)))
        classic_state = envi.open(classic_dir + 'output/{}_state.hdr'.format(instrument_tag)).open_memmap(interleave='bip')[:,:,-2:]
        classic_uncert = envi.open(classic_dir + 'output/{}_uncert.hdr'.format(instrument_tag)).open_memmap(interleave='bip')[:,:,-2:]

        # Load loc information
        loc_filepath = './data/testsim_full_loc_obs/ang20210710t100946_testsim_full_loc.hdr'
        X = envi.open(loc_filepath).open_memmap(interleave='bip')[...,:-1].copy()

        #GPR bounds
        # length_scale_b = np.array([[1e-4, 1e-1],[1e-7,1e-2]]) #AOD, water vapor
        # sigmaf_b = np.array([[1e-4,1e0],[1e-2,1e3]]) #AOD, water vapor

        #v5bounds
        length_scale_b = np.array([[1e-3, 1e-1],[1e-5,1e-3]]) #AOD, water vapor #np.array([[1e-4, 1e-1],[1e-8,1e-5]]) #AOD, water vapor
        sigmaf_b = np.array([[1e-3,1e0],[1e-3,1e2]]) #AOD, water vapor
        # #gpr_uncert = np.array([0.01,0.01]) #AOD, water vapor


        #List number of points to sample
        n_pts_list = np.array([5550,4150,1650,850,350,150,50,15]) #30 seg instead of 20
        #Initial points investigated: [8350,4150,1650,850,350,150,50,15]
        #Roughly equivalent to segment sizes of [20,40,100,200,500,1000,5000,10000] for images of the same size

        rmse_pts = np.zeros((len(n_pts_list),2))
        l_opt_pts = rmse_pts.copy()
        sigmaf_opt_pts = rmse_pts.copy()

        true_atm = true_state


        for jj in range(len(n_pts_list)):
            n_pts = n_pts_list[jj]
            print(f'Processing GPR drawn from {n_pts}...')

            #Start with a given random seed so can debug
            rn_generator = np.random.default_rng(20220531)
            x_gen = rn_generator.integers(0,X.shape[0],size=n_pts)
            y_gen = rn_generator.integers(0,X.shape[1],size=n_pts)

            #Get: locs, states, uncert for each point
            #Shape should be (n_pts,2)
            X_sub = X[x_gen,y_gen,:].copy()
            state_input  = classic_state[x_gen,y_gen,:]
            uncert_input = classic_uncert[x_gen,y_gen,:]


            #Run GPR for each

            #Save GPR, l, sigma f info

            #Compare to ground truth; save RMSE

            gpr_atm         = np.zeros(true_atm.shape)
            l_opt_save      = np.zeros(true_atm.shape[-1])
            sigmaf_opt_save = l_opt_save.copy()

            restarts = 20 #00  # number of times we want to restart the minimzation
            init_opt = np.zeros((restarts,2,true_atm.shape[-1]))
            cost_opt = np.zeros((restarts,true_atm.shape[-1]))
            l_opt_all      = cost_opt.copy()
            sigmaf_opt_all = cost_opt.copy()
            n_iter_opt     = cost_opt.copy()

            for _s in range(true_atm.shape[-1]):

                print(f'state{_s}')
                A_sub = state_input[:,_s].flatten()

                A_uncert_sub = uncert_input[:,_s] #0.001*np.ones(A_sub.shape) #Small uncertainty
                cos_sim_sub  = 0 # No additional correlated error guesstimates

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
                init = np.power(10,np.random.rand(restarts, len(b_log10)) * (b_log10[:, 1] - b_log10[:, 0]) + b_log10[:, 0]) #Make sure we're initializing from across orders of magnitude
                best = np.inf

                init_opt[:,:,_s] = init
                print('iter: ',end='')
                for i in range(restarts):
                    print(f'{i}',end=',')
                    res = minimize(nll_fn(X_sub, A_sub - np.mean(A_sub), neg_half_sqdist, A_uncert_sub, cos_sim_sub),
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

                # Store the optimization results in global variables for prediction with GP
                #l_opt, sigma_f_opt, sigma_cos_opt = res.x
                l_opt, sigma_f_opt = res.x
                sigma_cos_opt = 1
                #l_opt, sigma_f_opt, sigma_cos_opt = 0.02, 0.2, 0.2

                l_opt_save[_s] = l_opt
                sigmaf_opt_save[_s] = sigma_f_opt

                # make prediction with GP for each pixel in the scene
                # A_pred, A_cov = posterior(X_input.reshape(-1, 2), X_sub, A_sub - np.mean(A_sub), cos_sim_sub,
                #                    l=l_opt, sigma_f=sigma_f_opt, sigma_y=A_uncert_sub,sigma_cos=sigma_cos_opt)

                A_pred = posterior(X.reshape(-1, 2), X_sub, A_sub - np.mean(A_sub), cos_sim_sub,
                               l=l_opt, sigma_f=sigma_f_opt, sigma_y=A_uncert_sub,sigma_cos=sigma_cos_opt)

                A_shape = X.shape[0:2]

                gpr_atm[:,:,_s] = A_pred.reshape(A_shape) + np.mean(A_sub)    


            mdict = {'gpr_atm': gpr_atm,
                     'l_opt_all': l_opt_all,
                     'sigmaf_opt_all': sigmaf_opt_all,
                     'cost_opt': cost_opt,
                     'n_iter_opt': n_iter_opt,
                     'l_opt_save': l_opt_save,
                     'sigmaf_opt_save': sigmaf_opt_save,
                     'init_opt': init_opt}
            scio.savemat(retrieval_dir_list[0] + f'classic{output_tag_list[kk]}_gpr_v5b_n{n_pts}_v0.mat',mdict)

            rmse_pts[jj,:]=np.squeeze(np.sqrt(np.mean((gpr_atm-true_atm)**2,axis=(0,1))))
            l_opt_pts[jj,:] = l_opt_save
            sigmaf_opt_pts[jj,:] = sigmaf_opt_save

        mdict = {'n_pts_list': n_pts_list,
                 'l_opt_pts':  l_opt_pts,
                 'sigmaf_opt_pts': sigmaf_opt_pts,
                 'rmse_pts':   rmse_pts}

        scio.savemat(retrieval_dir_list[0] + f'classic{output_tag_list[kk]}_gpr_v5b_all_v0.mat',mdict)   
        
        
if __name__ == "__main__":
    main()