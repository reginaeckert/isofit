# Files to read in for simulation

obs=./data/testsim_full_loc_obs/ang20210710t100946_testsim_full_obs
loc=./data/testsim_full_loc_obs/ang20210710t100946_testsim_full_loc
wl=./data/wavelengths.txt
surface=./data/emit_surface.mat
lut_config=./lut/simulation_ang20210710t100946_plain_lut_config-nopresolve-aodinit.json


data_dir=./data/
rdn_base=simplain_full_raw
data_tag=ang20210710t100946
output_tag='_v5b_uncEmp_aodinit0.25'
output_dir_base=./retrievals/scoe_retrievals

# Loop through experiments (saved as different files)
for exp in indiawater_gaussaod0.2 indiawater_gaussaod0.2_noise2.0inst indiawater_gaussaod0.2_noise5.0inst indiawater_gaussaod0.2_noise10.0inst #indiawater_gaussaod0.2_noise1.0inst
do
  # Loop through covariance matrix estimation / error methods
  for em in noerr
  #for em in ce cosdis cosdis_norm pca_cosdis pca_cosdis_norm
  do
    mkdir ${output_dir_base}_${em}
    # Loop through segmentation sizes
    #for segsize in 20000 10000 5000 40 100 200
    for segsize in 500 1000
    do          
        # Set up server batch call to apply_oe
        sbatch -N 1 -c 40 -o logs/scoe_${exp}_${em}_${segsize}_o -e logs/scoe_${exp}_${em}_${segsize}_e --mem=180G --job-name=${exp}_${segsize} --wrap="python SC_OE.py ${data_dir}${data_tag}_${rdn_base}_${exp}/${data_tag}_${rdn_base}_${exp}_simulated_rdn $loc $obs ${output_dir_base}_${em}/${data_tag}_${rdn_base}_${exp}${output_tag}_${segsize} ang --surface_path=${surface} --log_file=${output_dir_base}_${em}/${exp}${output_tag}_${segsize}_runlog.txt --n_cores=40 --wavelength_path=$wl  --emulator_base=../../emulator/models/emulatordim_-2_slice_0 --segmentation_size=${segsize} --err_matrix=$em --lut_config_file=${lut_config} --presolve=0 --gpr_uncert=-1 --flag_run_part1_only --flag_skip_inference"
    done
  done
done

