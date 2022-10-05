# Forward simulate radiance from a reflectance cube, water vapor, and AOD map
# using ISOFIT infrastructure
# Originally written by Phil Brodrick, 2021
# Edited by Regina Eckert, October 2022, reckert@jpl.nasa.gov

import numpy as np
import pandas as pd
from spectral.io import envi
from isofit.radiative_transfer.modtran import ModtranRT
from isofit.core.common import resample_spectrum
import re
from scipy import interpolate
import scipy.interpolate as spi
from isofit.core.geometry import Geometry
from isofit.core.forward import ForwardModel
from isofit.configs import configs
import ray
import os
import argparse
from osgeo import gdal
from scipy import ndimage
import warnings
import time
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Forward simulate radiance.")


parser.add_argument('--water',type=str,default='2')
parser.add_argument('--output_base', type=str,default='testsim')
parser.add_argument('--output_tag', type=str,default='')
parser.add_argument('--data_dir', type=str,default='./data/')
parser.add_argument('--config_file',type=str,default='simulation_ang20210710t100946_config.json')
parser.add_argument('--aod_offset',type=float,default=0.2)
parser.add_argument('--flag_save_loc_obs', action='store_true')
parser.add_argument('--flag_run_small', action='store_true')
parser.add_argument('--flag_smooth', action='store_true')
parser.add_argument('--flag_instrument_noise', action='store_true')
parser.add_argument('--noise_factor',type=float,default=0)
parser.add_argument('--noise_file',type=str,default='./data/avirisng_noise.txt')

args = parser.parse_args()

def main(args): 
    
    start_time = time.time()
        
    # Switched to use retrievals generated under strong surface priors through classic
    # (pixel by pixel) retrieval to eliminate mismatches between our original scene and 
    # our surface priors due to water vapor residual features
    # Should only run recon with a looser prior (EMIT SDS) for this data
    rfl_file = './data/fwd_sim_base/ang20210710t100946_simgndalt_full_smooth_indiawater_gaussaod0.2_classicRetrieval/ang20210710t100946_rfl'
    obs_file = './data/testsim_full_loc_obs/ang20210710t100946_testsim_full_obs' 
    loc_file = './data/testsim_full_loc_obs/ang20210710t100946_testsim_full_loc'
    # *** Need to make sure obs, loc size match the rfl file data size (they use the same indices)

    water_file = './data/india_water.tif'
    india_water = gdal.Open(water_file, gdal.GA_ReadOnly).ReadAsArray()
    
    # Define and create output files
    exp_tag = '{}water_gaussaod{}'.format(args.water,args.aod_offset)
    data_tag = 'ang20210710t100946'
    
    # Flag-controlled tags
    size_tag = 'small' if args.flag_run_small else 'full'
    smooth_tag = 'smooth' if args.flag_smooth else 'raw'
    noise_tag = 'noise{}'.format(args.noise_factor) + ('inst' if args.flag_instrument_noise else 'const')
    
    output_tag = '{}_{}_{}_{}_{}_{}{}'.format(data_tag,args.output_base,size_tag,smooth_tag,exp_tag,noise_tag,args.output_tag)
    
    output_dir = output_tag + '/'
    print('Output directory: {}'.format(output_dir))
    if not os.path.exists(args.data_dir + output_dir):
        os.makedirs(args.data_dir + output_dir)

    output_rdn_file = args.data_dir + output_dir + output_tag + '_simulated_rdn'
    output_state_file = args.data_dir + output_dir + output_tag + '_true_state'
    
    print('Output files:\n{}\n{}'.format(output_state_file,output_rdn_file))

    # Set up row, column index into data
    # base_rows = [2000,2000+396] #Original OG data, also needed for obs, loc
    # base_cols = [100,100+420]   
    base_rows = [0,396]
    base_cols = [0,420]
    if args.flag_run_small:
        offset_rc = [90, 200]
        size_rc = [20,20]
        rows = [base_rows[0]+offset_rc[0], base_rows[0]+offset_rc[0]+size_rc[0]]
        cols = [base_cols[0]+offset_rc[1], base_cols[0]+offset_rc[1]+size_rc[1]]
    else:
        rows = base_rows
        cols = base_cols
            
    # Open files to save into
    meta = envi.open(rfl_file + '.hdr').metadata.copy()
    del meta['map info']
    meta['lines'] = rows[1] - rows[0]
    meta['samples'] = cols[1] - cols[0]

    output_rdn_ds = envi.create_image(output_rdn_file+'.hdr', meta, ext='', force=True)
    output_rdn = output_rdn_ds.open_memmap(interleave='bip',writable=True)

    meta['bands'] = int(meta['bands']) + 2
    output_state_ds = envi.create_image(output_state_file+'.hdr', meta, ext='', force=True)
    output_state = output_state_ds.open_memmap(interleave='bip',writable=True)
    
    if args.flag_save_loc_obs:
        output_loc_obs_dir = args.output_base + '_' + size_tag + '_loc_obs/'
        print('Output directory for loc &  obs files: {}'.format(output_loc_obs_dir))
        if not os.path.exists(args.data_dir + output_loc_obs_dir):
            os.makedirs(args.data_dir + output_loc_obs_dir)
        
        #Save smaller obs and loc files
        output_loc_file = args.data_dir + output_loc_obs_dir + data_tag + '_' + args.output_base + '_' + size_tag + '_loc'
        output_obs_file = args.data_dir + output_loc_obs_dir + data_tag + '_' + args.output_base + '_' + size_tag + '_obs'

        meta = envi.open(obs_file + '.hdr').metadata.copy()
        meta['lines'] = rows[1] - rows[0]
        meta['samples'] = cols[1] - cols[0]

        output_obs_ds = envi.create_image(output_obs_file+'.hdr', meta, ext='', force=True)
        output_obs = output_obs_ds.open_memmap(interleave='bip',writable=True)

        meta = envi.open(loc_file + '.hdr').metadata.copy()
        meta['lines'] = rows[1] - rows[0]
        meta['samples'] = cols[1] - cols[0]

        output_loc_ds = envi.create_image(output_loc_file+'.hdr', meta, ext='', force=True)
        output_loc = output_loc_ds.open_memmap(interleave='bip',writable=True)
    
    print('Configuring run...')
    
    # Sets up config information
    # Then runs initial forward model set-up to make sure all the files are in place
    # This avoids doing some set-up (creating pkl files etc) when in parallel operation
    # config_file = args.data_dir + 'simulation_ang20210710t100946_config.json' #OG data
    config_file = args.data_dir + args.config_file
    config = configs.create_new_config(config_file)
    ray.init()
    fm = ForwardModel(config)
    del fm
    ray.shutdown()
    
    print('Setting up AOD distribution with offset {}...'.format(args.aod_offset))
    
    # * Set up a Gaussian image, then crop and scale it to use as AOD map
    base_size = np.array([base_rows[1]-base_rows[0],base_cols[1]-base_cols[0]]); #"Full size" simulation shape for consistent Gaussian AOD 
    # base_size = np.array([396,420]); #"Full size" simulation shape for consistent Gaussian AOD (debugging)
    x, y = np.meshgrid(np.linspace(-1,1,base_size[1]*5), np.linspace(-1,1,base_size[0]*5))

    #Create gaussian
    d = np.sqrt(x*x+y*y)
    sigma, mu = 4, 0
    aod = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    aod[aod < 0] = 0
    #aod = aod[:output_state.shape[0], :output_state.shape[1]]
    aod = aod[:base_size[0], :base_size[1]]
    aod = aod - np.min(aod)
    aod = aod / np.max(aod)
    aod = aod / 10 + args.aod_offset
    
    if args.flag_run_small:
        aod = aod[offset_rc[0]:offset_rc[0]+size_rc[0],offset_rc[1]:offset_rc[1]+size_rc[1]]
    
    #Find good and bad wavelength ranges
    good_bands = np.loadtxt(args.data_dir + '/good_bands.txt').astype(np.int8)
    nan_bands = good_bands.copy().astype(float)
    nan_bands[np.logical_not(good_bands)] = np.nan
    bad_bands = np.logical_not(good_bands)
    
    # * Set up state vector 
    # Looping through data rows (original values are 2000 to 2395) which is a spatial dimension
    # Cols is also a spatial direction (crosstrack direction -> short direction across overall image)
    
    if args.flag_smooth:
        smooth_sigma = 1.5
        print_tag = 'smoothed reflectance (sigma = {}) and '.format(smooth_sigma)
    else:
        print_tag = ''
        
    print('Creating state vector with {}water {}...'.format(print_tag,args.water))

    for _r in range(rows[0], rows[1]):
        # * Reads in simulation reflectance data
        # Data is a 2D slice into 3D volume, output is from one spatial row and is:
        # [spatial column, spectral channel]
        rfl = envi.open(rfl_file + '.hdr').open_memmap(interleave='bip')[_r,cols[0]:cols[1],:] 
        
        st = np.zeros((rfl.shape[0], rfl.shape[1] + 2)) # Append two elements to state vector [AOD, water vapor]

        if args.flag_smooth:
            # Smooth the reflectance to get rid of some noise
            rfl_smooth = ndimage.gaussian_filter1d(rfl,sigma=smooth_sigma,axis=1)

            # Need to replace smoothed areas that have contributions from the bad bands
            good_bands_smooth = ndimage.gaussian_filter1d(good_bands.astype(float),sigma=smooth_sigma,axis=0)
            replace_bands = np.logical_and(good_bands_smooth < 1.0,good_bands_smooth > 0.0) #Find where contrib are
            rfl_smooth[:,replace_bands] = rfl[:,replace_bands] # Replace it with the unsmoothed data
            
            st[:,:-2] = rfl_smooth  
        else:
            st[:,:-2] = rfl 


        # * Reads in water vapor map from India campaign
        # * Also have option to set args.water = constant value instead
        if args.water == 'india':
            st[:,-1] = india_water[_r - rows[0],:] #First row of this dataset, subtract offset
        else:
            st[:,-1] = float(args.water) #Last value in state vector

        # * Append AOD to the the state vector
        st[:, -2] = aod[_r - rows[0], :]  #First row of this dataset, subtract offset 
        #2nd to last value in state vector

        # State vector for each [row,col] spatial location is [spectrum, AOD, water vapor (column)]

        output_state[_r- rows[0],...] = st.copy() # Save it in the file for this row (all columns in row)

    # Data was written to a file, so we don't need to keep it in memory
    del output_rdn, output_state, output_rdn_ds, output_state_ds
    
    # * Call forward model on each line 
    # --> Line appears to be a spatial line (one spatial row of image)
    # * Output is radiance value
    # *** Absolutely should have this defined outside main and pass in values instead of having them drop through....
    
    # * Instatiate forward model
    fm = ForwardModel(config)
    
    #Set up noise
    if args.noise_factor > 0:
        flag_add_noise = True
        print('Adding noise')
        if args.flag_instrument_noise:
            print('Pulling parametric noise coeffs from ' + args.noise_file)
            #Load in file
            coeffs = np.loadtxt(args.noise_file, delimiter=' ', comments='#')
            p_a, p_b, p_c = [spi.interp1d(coeffs[:, 0], coeffs[:, col],fill_value='extrapolate') for col in (1, 2, 3)]
            parametric_noise = np.array([[p_a(w), p_b(w), p_c(w)] for w in fm.instrument.wl_init])       
    else:
        flag_add_noise = False
        print('Noiseless simulation')
        
    @ray.remote
    def simulate_line(line):
        # * Load in data
        #print(output_state_file)
        state = envi.open(output_state_file + '.hdr').open_memmap(interleave='bip')
        obs = envi.open(obs_file + '.hdr').open_memmap(interleave='bip')[rows[0]:rows[1],cols[0]:cols[1],:]
        loc = envi.open(loc_file + '.hdr').open_memmap(interleave='bip')[rows[0]:rows[1],cols[0]:cols[1],:]
        output_rdn = envi.open(output_rdn_file + '.hdr').open_memmap(interleave='bip',writable=True)
        if args.flag_save_loc_obs:
            output_loc = envi.open(output_loc_file + '.hdr').open_memmap(interleave='bip',writable=True)
            output_obs = envi.open(output_obs_file + '.hdr').open_memmap(interleave='bip',writable=True)
        # ** Strange that rows, cols do not look like they would be instantiated here....
        # * ^ Because the function is defined within the outer function, rather than outside (same namespace)

        # * Instatiate forward model
        #fm = ForwardModel(config)
        print(line)
        for _c in range(state.shape[1]):
            # * For each column --> giving *just* the spectral dimension (+ atmospheric states)
            # * Get the geometry (isofit>>Geometry)
            geom = Geometry(obs=obs[line,_c,:], loc=loc[line,_c,:])
            # * Copy the state vector
            st = state[line,_c,:].copy() # Information from one spatial pixel
            # * Calculate the output radiance
            rdn = fm.calc_rdn(st, geom) # Presumably this means fm.calc_rdn expects water vapor, AOD as part of the state vector...
            
            if flag_add_noise:
                if args.flag_instrument_noise:
                    noise_plus_meas = parametric_noise[:, 1] + rdn
                    if np.any(noise_plus_meas <=0):
                        noise_plus_meas[noise_plus_meas <= 0] = 1e-5
                        print('Parametric noise model found noise <= 0 - adjusting to slightly positive to avoid /0.')
                    nedl = np.abs(parametric_noise[:, 0]*np.sqrt(noise_plus_meas)+parametric_noise[:, 2])
                    mu   = np.zeros(rdn.shape)
                    Sy   = np.diagflat(np.power(nedl*args.noise_factor,2))
                    rdn  = rdn + np.random.multivariate_normal(mu, Sy)
                else:
                    rdn  = rdn + args.noise_factor*np.random.standard_normal(rdn.shape)
                 
            # * Save it
            output_rdn[line,_c,:] = rdn
            if args.flag_save_loc_obs:
                output_loc[line,_c,:] = loc[line,_c,:]
                output_obs[line,_c,:] = obs[line,_c,:]

        # * Since saving everything to ENVI file, can then delete it
        del state, obs, loc, output_rdn
        
        if args.flag_save_loc_obs:
            del output_loc, output_obs
     
    print('Running forward model...')
    
    # * Actual call of forward model runs
    ray.init(num_cpus=40)
    # Loop through each row
    # Call simulate_line for each [col, spectrum] "line"
    jobs = [simulate_line.remote(_r) for _r in range(aod.shape[0])] 
    rreturn = [ray.get(jid) for jid in jobs]
    ray.shutdown()
    
    end_time = time.time()
    print('Total time elapsed: {} s'.format(end_time-start_time))


if __name__ == '__main__':
    main(args)