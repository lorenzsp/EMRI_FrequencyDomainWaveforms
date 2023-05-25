# nohup python emri_pe.py -Tobs 2.0 -M 1e6 -mu 10.0 -p0 12.0 -e0 0.35 -dev 6 -eps 1e-3 -dt 10.0 -injectFD 0 -template fd -nwalkers 32 -ntemps 2 > out1.out &
# nohup python emri_pe.py -Tobs 2.0 -M 1e6 -mu 10.0 -p0 12.0 -e0 0.35 -dev 7 -eps 1e-3 -dt 10.0 -injectFD 1 -template fd -nwalkers 32 -ntemps 2 -downsample 1 > out4.out &
# nohup python emri_pe.py -Tobs 2.0 -M 1e6 -mu 10.0 -p0 12.0 -e0 0.35 -dev 4 -eps 1e-3 -dt 10.0 -injectFD 1 -template fd -nwalkers 32 -ntemps 2 > out3.out &
# nohup python check_mode_by_mode.py -Tobs 4.0 -dev 4 -eps 1e-5 -dt 10.0 > outcheck3.out &
# nohup python check_mode_by_mode.py -Tobs 4.0 -dev 4 -eps 1e-5 -dt 10.0 > outcheck3.out &


# Define an array of values for Tobs, eps, and dt
#!/bin/bash


# Iterate over the values
Tobs=4
eps=1e-5
dt=5.0
output_file="outcheck_Tobs${Tobs}_eps${eps}_dt${dt}.out"
nohup python check_mode_by_mode.py -Tobs $Tobs -dev 1 -eps $eps -dt $dt > "$output_file" &

Tobs=4
eps=1e-5
dt=5.0
output_file="outcheck_Tobs${Tobs}_eps${eps}_dt${dt}_fixed_insp0.out"
nohup python check_mode_by_mode.py -Tobs $Tobs -dev 2 -eps $eps -dt $dt -fixed_insp 0 > "$output_file" &

Tobs=4
eps=1e-5
dt=10.0
output_file="outcheck_Tobs${Tobs}_eps${eps}_dt${dt}.out"
nohup python check_mode_by_mode.py -Tobs $Tobs -dev 3 -eps $eps -dt $dt > "$output_file" &

Tobs=4
eps=1e-2
dt=5.0
output_file="outcheck_Tobs${Tobs}_eps${eps}_dt${dt}.out"
nohup python check_mode_by_mode.py -Tobs $Tobs -dev 4 -eps $eps -dt $dt > "$output_file" &

Tobs=2
eps=1e-5
dt=5.0
output_file="outcheck_Tobs${Tobs}_eps${eps}_dt${dt}.out"
nohup python check_mode_by_mode.py -Tobs $Tobs -dev 5 -eps $eps -dt $dt > "$output_file" &

# # Define an array of values for Tobs, eps, and dt
# Tobs_values=(4)
# eps_values=(1e-2)
# dt_values=(1.0)
# for Tobs in "${Tobs_values[@]}"; do
#   for eps in "${eps_values[@]}"; do
#     for dt in "${dt_values[@]}"; do
#       # Create a unique string for the output file
#       output_file="outcheck_Tobs${Tobs}_eps${eps}_dt${dt}.out"

#       # Run Python code with current variable values
#       nohup python check_mode_by_mode.py -Tobs $Tobs -dev  -eps $eps -dt $dt > "$output_file" &
#     done
#   done
# done

