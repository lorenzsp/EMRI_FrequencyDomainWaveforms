# nohup python emri_pe.py > out.out &

import sys
sys.path.append("../LISAanalysistools/")
sys.path.append("../Eryn/")

import numpy as np
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
import corner
from lisatools.utils.utility import AET

from eryn.moves import StretchMove
from lisatools.sampling.likelihood import Likelihood
from lisatools.diagnostic import *

from lisatools.sensitivity import get_sensitivity

from few.waveform import GenerateEMRIWaveform
from few.utils.utility import get_mu_at_t
from eryn.utils import TransformContainer
import time

import matplotlib.pyplot as plt
from few.utils.constants import *
SEED=2601996
np.random.seed(SEED)

try:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(5)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

import warnings

warnings.filterwarnings("ignore")

# whether you are using 
use_gpu = True

if use_gpu is True:
    xp = np

if use_gpu and not gpu_available:
    raise ValueError("Requesting gpu with no GPU available or cupy issue.")

few_gen = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux", 
    sum_kwargs=dict(pad_output=True, output_type="fd"),
    use_gpu=use_gpu,
    return_list=False
)

td_gen = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux", 
    sum_kwargs=dict(pad_output=True),
    use_gpu=use_gpu,
    return_list=True
)

# conversion
class get_fd_waveform():

    def __init__(self, waveform_generator):
        self.waveform_generator = waveform_generator

    def __call__(self,*args, **kwargs):
        initial_out = self.waveform_generator(*args, **kwargs)
        # frequency goes from -1/dt/2 up to 1/dt/2
        self.frequency = self.waveform_generator.waveform_generator.create_waveform.frequency
        self.positive_frequency_mask = (self.frequency>=0.0)
        list_out = self.transform_FD(initial_out)
        return [list_out[0][self.positive_frequency_mask], list_out[1][self.positive_frequency_mask]]

    def transform_FD(self, input_signal):
        fd_sig = -xp.flip(input_signal)
        fft_sig_p = xp.real(fd_sig + xp.flip(fd_sig) )/2.0 + 1j * xp.imag(fd_sig - xp.flip(fd_sig))/2.0
        fft_sig_c = -xp.imag(fd_sig + xp.flip(fd_sig) )/2.0 + 1j * xp.real(fd_sig - xp.flip(fd_sig))/2.0
        return [fft_sig_p, fft_sig_c]


class get_fd_waveform_fromTD():

    def __init__(self, waveform_generator, positive_frequency_mask, dt):
        self.waveform_generator = waveform_generator
        self.positive_frequency_mask = positive_frequency_mask
        self.dt = dt

    def __call__(self,*args, **kwargs):
        data_channels_td = self.waveform_generator(*args, **kwargs)
        fft_td_wave_p = xp.fft.fftshift(xp.fft.fft(data_channels_td[0]))[self.positive_frequency_mask] * self.dt
        fft_td_wave_c = xp.fft.fftshift(xp.fft.fft(data_channels_td[1]))[self.positive_frequency_mask] * self.dt
        return [fft_td_wave_p,fft_td_wave_c]

# function call
def run_check(
    Tobs,
    dt,
    fp,
    ntemps,
    nwalkers,
    injectFD=1,
    template='fd',
    emri_kwargs={},
):

    # for transforms
    # this is an example of how you would fill parameters 
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
       "ndim_full": 14,
       "fill_values": np.array([0.0, 0.0, 0.0]), # spin and inclination and Phi_theta
       "fill_inds": np.array([2, 5, 12]),
    }

    # priors
    priors = {
        "emri": ProbDistContainer(
            {
                0: uniform_dist(np.log(5e5), np.log(5e6)),  # M
                1: uniform_dist(1.0, 100.0),  # mu
                2: uniform_dist(10.0, 15.0),  # p0
                3: uniform_dist(0.001, 0.7),  # e0
                4: uniform_dist(0.01, 100.0),  # dist in Gpc
                5: uniform_dist(-0.99999, 0.99999),  # qS
                6: uniform_dist(0.0, 2 * np.pi),  # phiS
                7: uniform_dist(-0.99999, 0.99999),  # qK
                8: uniform_dist(0.0, 2 * np.pi),  # phiK
                9: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                10: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            }
        ) 
    }

    # sampler treats periodic variables by wrapping them properly
    periodic = {
        "emri": {6: 2 * np.pi, 8: np.pi, 9: 2 * np.pi, 10: 2 * np.pi}
    }

    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
    parameter_transforms = {
        0: np.exp,  # M 
        7: np.arccos, # qS
        9: np.arccos,  # qK
    }

    transform_fn = TransformContainer(
        parameter_transforms=parameter_transforms,
        fill_dict=fill_dict,

    )

    from few.trajectory.inspiral import EMRIInspiral
    traj_module = EMRIInspiral(func="SchwarzEccFlux")

    factor = []
    overlap = []
    failed_points = []
    for el in range(500):
        print('---------------------')
        tmp = priors["emri"].rvs()
        # get injected parameters after transformation
        injection_in = transform_fn.both_transforms(tmp)[0]

        # set initial parameters
        M = injection_in[0]
        p0 = injection_in[3]
        e0 = injection_in[4]

        traj_args = [M, 0.0, p0, e0, 1.0]
        traj_kwargs = {}
        index_of_mu = 1

        t_out = Tobs*1.001
        try:
            # run trajectory
            injection_in[1] = get_mu_at_t(
                traj_module,
                t_out,
                traj_args,
                index_of_mu=index_of_mu,
                traj_kwargs=traj_kwargs,
                xtol=2e-12,
                rtol=8.881784197001252e-16,
                # bounds=[6+2*e0+0.2, 16.0],
            )

            tic = time.perf_counter()
            # generate FD waveforms
            data_channels_fd = few_gen(*injection_in, **emri_kwargs)
            # frequency goes from -1/dt/2 up to 1/dt/2
            frequency = few_gen.waveform_generator.create_waveform.frequency
            positive_frequency_mask = (frequency>=0.0)
            fd_gen = get_fd_waveform(few_gen)
            # transform into hp and hc
            sig_fd = fd_gen.transform_FD(data_channels_fd)
            toc = time.perf_counter()
            fd_time = toc-tic

            tic = time.perf_counter()
            # generate TD waveform, this will return a list with hp and hc
            data_channels_td = td_gen(*injection_in, **emri_kwargs)
            # fft from negative to positive frequencies
            fft_td_wave_c = xp.fft.fftshift(xp.fft.fft(data_channels_td[1])) * dt
            fft_td_wave_p = xp.fft.fftshift(xp.fft.fft(data_channels_td[0])) * dt
            sig_td = [fft_td_wave_p,fft_td_wave_c]
            toc = time.perf_counter()
            td_time = toc-tic
            print("TD/FD time",td_time/fd_time, "TD", td_time, "FD", fd_time )
            if el>0:
                factor.append(td_time/fd_time)
            # kwargs for computing inner products
            fd_inner_product_kwargs = dict( PSD="cornish_lisa_psd", use_gpu=use_gpu, f_arr=frequency[positive_frequency_mask])
            sig_fd = [sig_fd[0][positive_frequency_mask],sig_fd[1][positive_frequency_mask]]
            sig_td = [sig_td[0][positive_frequency_mask],sig_td[1][positive_frequency_mask]]

            Over = np.abs(1-inner_product(sig_fd, sig_td, normalize=True, **fd_inner_product_kwargs))
            print("Overlap total and partial ", Over)
            overlap.append(Over.get())

        except:
            failed_points.append(injection_in)
            print("not found for params",tmp[:3])
    
    plt.figure()
    plt.hist(factor,bins=25)
    plt.xlabel('TD/FD speed up factor')
    plt.savefig(fp + 'hist_timing.png')

    overlap = np.asarray(overlap)
    plt.figure()
    plt.hist(np.log10(overlap),bins=25)
    plt.xlabel('log10 Mismatch')
    plt.savefig(fp + 'overlap.png')
    
    np.save(fp +"failed_points.npy",np.asarray(failed_points))
    
    
    return

if __name__ == "__main__":

    Tobs = 2.05
    dt = 10.0
    eps = 1e-2

    ntemps = 4
    nwalkers = 30

    waveform_kwargs = {
        "T": Tobs,
        "dt": dt,
        "eps": eps
    }

    fp = f"emri_T{Tobs}_seed{SEED}_dt{dt}_eps{eps}_"

    run_check(
        Tobs,
        dt,
        fp,
        ntemps,
        nwalkers,
        emri_kwargs=waveform_kwargs
    )
