# EMRI_FourierDomainWaveforms

This folder contains the analysis of the Fourier Domain EMRI Waveform implemented in the package Fast EMRI Waveforms ([arxiv.org/2104.04582](https://arxiv.org/abs/2104.04582), [arxiv.org/2008.06071](https://arxiv.org/abs/2008.06071)). To run the scripts constained in this folder it is necessary to install the Fast EMRI Waveforms package:

```
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
cd FastEMRIWaveforms/
git checkout fd
```

```
conda create -n fd_env -c conda-forge clangxx_osx-64 clang_osx-64 wget gsl lapack=3.6.1 hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib python=3.10
y
conda activate fd_env
pip install tqdm corner

python setup.py install --no_omp
python -m unittest discover
```



```
export PATH=$PATH:/usr/local/cuda-12.1/bin/
conda activate fd_env
pip install cupy-cuda12x
```

# Info downsampled posterior
python emri_pe.py -Tobs 4.0 -M 3670041.7362535275 -mu 292.0583167470244 -p0 13.709101864726545 -e0 0.5794130830706371 -dev 5 -eps 1e-2 -dt 10.0 -injectFD 1 -template fd -nwalkers 32 -ntemps 2 -downsample 100 --window_flag 0 
new p0  13.709101864726545
fd time 2.73667973279953
get_convolution time 4.22872613184154 length of signal 12623261
get_fft_td_windowed time 5.9704997930675745 length of signal 12623261
td time 15.866921707056463
shape (6311631,) (6311631,)
Overlap total and partial  0.9963315697438266 0.9963952508469567 0.9962679005730234
frequency len 12623261  make sure that it is odd
last point in TD 0.0
SNR =  78.75644717209373
Running with downsampling, injecing consistently the FD signal
---------------------------
--------------------------
downsampling  100
number of frequencies 3554
percentage of frequencies used 0.0005630874174995338
fd time 0.30734901564816636
SNR =  79.51725330664453
downsampled likelihood 0.3010125160217285
standard likelihood 2.8062386512756348
[0.43615124]
downsampled likelihood 0.21603989601135254
standard likelihood 2.742187023162842
[0.08587572]
downsampled likelihood 0.21008753776550293
standard likelihood 2.823539972305298
[0.04575854]
downsampled likelihood 0.20734310150146484
standard likelihood 2.8350822925567627
[0.20948247]
downsampled likelihood 0.20625638961791992
standard likelihood 2.8284435272216797
[0.0624068]
downsampled likelihood 0.21138691902160645
standard likelihood 2.8080475330352783
[0.66431942]
downsampled likelihood 0.22159934043884277
standard likelihood 2.8221259117126465
[0.12707666]
downsampled likelihood 0.2028636932373047
standard likelihood 2.7910027503967285
[0.20298264]
downsampled likelihood 0.20400261878967285
standard likelihood 2.7930784225463867
[0.21415242]
downsampled likelihood 0.20317935943603516
standard likelihood 2.7207674980163574