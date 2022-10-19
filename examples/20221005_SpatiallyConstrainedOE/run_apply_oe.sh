#Input file examples
unique_output_name=your_unique_output_name_here
base_dir=your_base_directory_here

input_radiance_file=${base_dir}/data/ang20210710t100946_simplain_full_raw_indiawater_gaussaod0.2_noise1.0inst/ang20210710t100946_simplain_full_raw_indiawater_gaussaod0.2_noise1.0inst_simulated_rdn
input_loc_file=${base_dir}/data/testsim_full_loc_obs/ang20210710t100946_testsim_full_loc
input_obs_file=${base_dir}/data/testsim_full_loc_obs/ang20210710t100946_testsim_full_obs

save_base_dir=${base_dir}/output/${unique_output_name}
log_file=${base_dir}/output/${unique_output_name}_runlog.txt
sensor_code=ang
surface_file=${base_dir}/data/emit_surface.mat
wavelength_path=${base_dir}/data/wavelengths.txt
lut_config_file=${base_dir}/data/simulation_ang20210710t100946_plain_lut_config-nopresolve.json

emulator_file=${base_dir}/emulator_files/sRTMnet_v100
#emulator_file is either path to folder containing .pb file (such as saved_model.pb) or the path to a .h5 file directly (such as sRTMnet_v110.h5)

#For run_fixed_isofit only
input_state=${base_dir}/data/ang20210710t100946_simplain_full_raw_indiawater_gaussaod0.2_noise1.0inst/ang20210710t100946_simplain_full_raw_indiawater_gaussaod0.2_noise1.0inst_true_state.hdr

#Basic pixel-by-pixel call structure - apply_oe
python3 apply_oe.py ${input_radiance_file} ${input_loc_file} ${input_obs_file} ${save_base_dir} ${sensor_code} --presolve=1 --empirical_line=0 --surface_path=${surface_file} --emulator_base=${emulator_file} --n_cores=40 --log_file=${log_file} --wavelength_path=${wavelength_path} --lut_config_file=${lut_config_file} (--multiple_restarts)
