
# Locations of data files
apply_oe=../isofit/isofit/utils/apply_oe.py
data_dir=./data/
loc=./data/testsim_full_loc_obs/ang20210710t100946_testsim_full_loc
obs=./data/testsim_full_loc_obs/ang20210710t100946_testsim_full_obs
wl=./data/wavelengths.txt
surface=./data/emit_surface.mat
lut_config=./lut/simulation_ang20210710t100946_plain_lut_config-nopresolve-aodinit.json

rdn_base=simplain_full_raw
data_tag=ang20210710t100946
output_tag='_aodinit0.25'
output_dir=./retrievals/classic_retrievals/

# Loop through experiments
#for exp in indiawater_gaussaod 2water_0.1aod 2water_gaussaod indiawater_0.1aod
for exp in indiawater_gaussaod0.2 #indiawater_gaussaod0.2_noise2.0inst indiawater_gaussaod0.2_noise5.0inst indiawater_gaussaod0.2_noise10.0inst #indiawater_gaussaod0.2_noise1.0inst
do
    # Set up batch server call to apply_oe to run optimal estimation recon from isofit
    sbatch -N 1 -c 40 -o logs/${exp}${output_tag}_o -e logs/${exp}${output_tag}_e --mem=180G --wrap="python ${apply_oe} ${data_dir}${data_tag}_${rdn_base}_${exp}/${data_tag}_${rdn_base}_${exp}_simulated_rdn $loc $obs ${output_dir}${data_tag}_${rdn_base}_${exp}${output_tag} ang --presolve=0 --empirical_line=0 --surface_path=${surface} --log_file=${output_dir}${rdn_base}${output_tag}_${exp}_runlog.txt --n_cores=40 --wavelength_path=$wl  --emulator_base=../../emulator/models/emulatordim_-2_slice_0 --lut_config_file=${lut_config}"
done


