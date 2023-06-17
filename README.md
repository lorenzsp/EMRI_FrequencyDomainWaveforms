# EMRI_FourierDomainWaveforms

This folder contains the analysis of the Fourier Domain EMRI Waveform implemented in the package Fast EMRI Waveforms ([arxiv.org/2104.04582](https://arxiv.org/abs/2104.04582), [arxiv.org/2008.06071](https://arxiv.org/abs/2008.06071)). To run the scripts constained in this folder it is necessary to install the Fast EMRI Waveforms package and then install the additional packages uding `pip install tqdm corner`.

```
cd FastEMRIWaveforms/
git checkout fd
python setup.py install
```



```
export PATH=$PATH:/usr/local/cuda-12.1/bin/
conda activate fd_env
pip install cupy-cuda12x
```

# Info downsampled posterior
skip every  235 th element
number of frequencies 26859
percentage of frequencies 0.004255476912386038

skip every  5165 th element
number of frequencies 1223
percentage of frequencies 0.000193769249184561