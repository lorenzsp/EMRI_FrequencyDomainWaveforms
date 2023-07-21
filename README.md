# EMRI Frequency Domain Waveforms

This folder contains the scripts to reproduce the Frequency Domain EMRI Waveform analysis presented in [arxiv-number-in-prep](). In `Tutorial_FrequencyDomain_Waveforms.ipynb` you can find how to generate Frequency Domain waveforms and compare them to the time domain ones as in the following figure

![time domain VS frequency domain](https://github.com/lorenzsp/EMRI_FourierDomainWaveforms/blob/main/figures/FD_TD_frequency.pdf?raw=true)

The Frequency Domain implementation is now part of the package Fast EMRI Waveforms ([arxiv.org/2104.04582](https://arxiv.org/abs/2104.04582), [arxiv.org/2008.06071](https://arxiv.org/abs/2008.06071)). To run the scripts contained in this folder it is necessary to install the Fast EMRI Waveforms package available [here](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/) with the following installation setup: `bash install.sh env_name=fd_env install_type=sampling`.

The analysis presented in [arxiv-number-in-prep]() can be reproduced using the scripts:
- emri_pe.py: runs an MCMC analysis of an EMRI source. Usage: 
    ```
    python emri_pe.py -Tobs 2.0 -M 1e6 -mu 10.0 -p0 12.0 -e0 0.35 -dev 7 -eps 1e-2 -dt 10.0 -injectFD 1 -template fd -nwalkers 16 -ntemps 1 -downsample 100
    ```
- check_mode_by_mode.py: scans the EMRI parameter space and compares the accuracy and performance of the time and frequency domain models. Usage: 
    ```
    python check_mode_by_mode.py -Tobs 1.0 -dev 5 -eps 1e-2 -dt 10.0 -fixed_insp 1 -nsteps 10
    ```
- Tutorial_FrequencyDomain_Waveforms.ipynb: describes the usage of the frequency domain waveform and compares it with the time domain
- Tutorial_FD_construction_single_mode.ipynb: describes the construction of the frequency domain waveform for a single harmonic

## Authors

* **Lorenzo Speri**
* Michael Katz
