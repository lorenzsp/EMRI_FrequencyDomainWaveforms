# EMRI_FourierDomainWaveforms

This folder contains the analysis of the Fourier Domain EMRI Waveform implemented in the package Fast EMRI Waveforms ([arxiv.org/2104.04582](https://arxiv.org/abs/2104.04582), [arxiv.org/2008.06071](https://arxiv.org/abs/2008.06071)). To run the scripts constained in this folder it is necessary to install the Fast EMRI Waveforms package, [Eryn](https://github.com/mikekatz04/Eryn) and [LISAanalysistools](https://github.com/mikekatz04/LISAanalysistools)

The analysis presented in [arxiv]() can be reproduced using the scripts:
- emri_pe.py: runs an MCMC analysis of an EMRI source
- check_mode_by_mode.py: scans the EMRI parameter space and compares accuracy and performance of the time and frequency domain models
- Tutorial_FrequencyDomain_Waveforms.ipynb: describes the usage of the frequency domain waveform and compares it with the time domain
- Tutorial_FD_construction_single_mode.ipynb: describes the construction of the frequency domain waveform for a single harmonic

The following notebooks illustrate how to use the frequency domain waveforms l
## Authors

* **Lorenzo Speri**
* Michael Katz
