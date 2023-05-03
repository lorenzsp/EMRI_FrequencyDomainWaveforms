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
from eryn.utils import TransformContainer
from few.utils.utility import omp_set_num_threads
omp_set_num_threads(8)

import matplotlib.pyplot as plt
from few.utils.constants import *
SEED=2601996
np.random.seed(SEED)

try:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(7)
    gpu_available = True
    use_gpu = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False
    use_gpu = False

import warnings
warnings.filterwarnings("ignore")

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
def run_emri_pe(
    emri_injection_params, 
    Tobs,
    dt,
    fp,
    ntemps,
    nwalkers,
    injectFD=1,
    template='fd',
    emri_kwargs={},
):

    # sets the proper number of points and what not
    
    N_obs = int(Tobs * YRSID_SI / dt) # may need to put "- 1" here because of real transform
    Tobs = (N_obs * dt) / YRSID_SI
    t_arr = xp.arange(N_obs) * dt

    # for transforms
    # this is an example of how you would fill parameters 
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
       "ndim_full": 14,
       "fill_values": np.array([0.0, 0.0, 0.0]), # spin and inclination and Phi_theta
       "fill_inds": np.array([2, 5, 12]),
    }

    (
        M,  
        mu,
        a, 
        p0, 
        e0, 
        x0,
        dist, 
        qS,
        phiS,
        qK, 
        phiK, 
        Phi_phi0, 
        Phi_theta0, 
        Phi_r0
    ) = emri_injection_params

    # get the right parameters
    # log of large mass
    emri_injection_params[0] = np.log(emri_injection_params[0])
    emri_injection_params[7] = np.cos(emri_injection_params[7]) 
    emri_injection_params[8] = emri_injection_params[8] % (2 * np.pi)
    emri_injection_params[9] = np.cos(emri_injection_params[9]) 
    emri_injection_params[10] = emri_injection_params[10] % (2 * np.pi)

    # phases
    emri_injection_params[-1] = emri_injection_params[-1] % (2 * np.pi)
    emri_injection_params[-2] = emri_injection_params[-2] % (2 * np.pi)
    emri_injection_params[-3] = emri_injection_params[-3] % (2 * np.pi)

    # remove three we are not sampling from (need to change if you go to adding spin)
    emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])

    # priors
    priors = {
        "emri": ProbDistContainer(
            {
                0: uniform_dist(np.log(5e5), np.log(5e6)),  # M
                1: uniform_dist(1.0, 100.0),  # mu
                2: uniform_dist(10.0, 15.0),  # p0
                3: uniform_dist(0.001, 0.5),  # e0
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

    # get injected parameters after transformation
    injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]

    # generate FD waveforms
    data_channels_fd = few_gen(*injection_in, **emri_kwargs)
    # frequency goes from -1/dt/2 up to 1/dt/2
    frequency = few_gen.waveform_generator.create_waveform.frequency
    positive_frequency_mask = (frequency>=0.0)
    # define converions class
    fd_gen = get_fd_waveform(few_gen)
    # transform into hp and hc
    sig_fd = fd_gen.transform_FD(data_channels_fd)

    # generate TD waveform, this will return a list with hp and hc
    data_channels_td = td_gen(*injection_in, **emri_kwargs)
    fft_td_gen = get_fd_waveform_fromTD(td_gen,positive_frequency_mask,dt)
    # fft from negative to positive frequencies
    fft_td_wave_c = xp.fft.fftshift(xp.fft.fft(data_channels_td[1])) * dt
    fft_td_wave_p = xp.fft.fftshift(xp.fft.fft(data_channels_td[0])) * dt
    sig_td = [fft_td_wave_p,fft_td_wave_c]
    # sig_td = fft_td_gen(*injection_in, **emri_kwargs)

    # kwargs for computing inner products
    fd_inner_product_kwargs = dict( PSD="cornish_lisa_psd", use_gpu=use_gpu, f_arr=frequency[positive_frequency_mask])
    sig_fd = [sig_fd[0][positive_frequency_mask],sig_fd[1][positive_frequency_mask]]
    sig_td = [sig_td[0][positive_frequency_mask],sig_td[1][positive_frequency_mask]]

    print("Overlap total and partial ", inner_product(sig_fd, sig_td, normalize=True, **fd_inner_product_kwargs),
    inner_product(sig_fd[0], sig_td[0], normalize=True, **fd_inner_product_kwargs),
    inner_product(sig_fd[1], sig_td[1], normalize=True, **fd_inner_product_kwargs)
    )

    check_snr = snr(sig_fd, **fd_inner_product_kwargs)
    print("SNR = ", check_snr)

    # this is a parent likelihood class that manages the parameter transforms
    nchannels = 2
    if template=='fd':
        like_gen = fd_gen
    elif template=='td':
        like_gen = fft_td_gen

    like = Likelihood(
        like_gen,
        nchannels,  # channels (plus,cross)
        parameter_transforms={"emri": transform_fn},
        vectorized=False,
        transpose_params=False,
        subset=1,  # may need this subset
        f_arr = frequency[positive_frequency_mask].get(),
        use_gpu=use_gpu
    )
    
    # inject a signal
    if bool(injectFD):
        data_stream = sig_fd
    else:
        data_stream = sig_td
    
    like.inject_signal(
        data_stream=data_stream,
        # params= injection_params.copy()[test_inds],
        waveform_kwargs=waveform_kwargs,
        noise_fn=get_sensitivity,
        noise_kwargs=dict(sens_fn="cornish_lisa_psd"),
        add_noise=False,
    )

    ndim = 11

    # generate starting points
    factor = 1e-5
    cov = np.ones(ndim) * 1e-4
    cov[0] = 1e-5

    start_like = np.zeros((nwalkers * ntemps))
    
    iter_check = 0
    max_iter = 1000
    while np.std(start_like) < 1.0:
        
        logp = np.full_like(start_like, -np.inf)
        tmp = np.zeros((ntemps * nwalkers, ndim))
        fix = np.ones((ntemps * nwalkers), dtype=bool)
        while np.any(fix):
            tmp[fix] = (emri_injection_params_in[None, :] * (1. + factor * cov * np.random.randn(nwalkers * ntemps, ndim)))[fix]

            emri_injection_params_in[5] = emri_injection_params_in[5] % (2 * np.pi)
            emri_injection_params_in[7] = emri_injection_params_in[7] % (2 * np.pi)

            # phases
            emri_injection_params_in[-1] = emri_injection_params_in[-1] % (2 * np.pi)
            emri_injection_params_in[-2] = emri_injection_params_in[-2] % (2 * np.pi)
            
            logp = priors["emri"].logpdf(tmp)

            fix = np.isinf(logp)
            if np.all(fix):
                breakpoint()

        # like.injection_channels[:] = 0.0
        start_like = like(tmp, **emri_kwargs)
    
        iter_check += 1
        factor *= 1.5

        print("std in likelihood",np.std(start_like))
        print("likelihood",start_like)

        if iter_check > max_iter:
            raise ValueError("Unable to find starting parameters.")

    start_params = tmp.copy()
    start_prior = priors["emri"].logpdf(start_params)

    # start state
    start_state = State(
        {"emri": start_params.reshape(ntemps, nwalkers, 1, ndim)}, 
        log_like=start_like.reshape(ntemps, nwalkers), 
        log_prior=start_prior.reshape(ntemps, nwalkers)
    )

    # MCMC moves (move, percentage of draws)
    moves = [
        StretchMove(use_gpu=use_gpu)
    ]

    # prepare sampler
    sampler = EnsembleSampler(
        nwalkers,
        [ndim],  # assumes ndim_max
        like,
        priors,
        tempering_kwargs={"ntemps": ntemps, "Tmax": np.inf},
        moves=moves,
        kwargs=emri_kwargs,
        backend=fp,
        vectorize=True,
        periodic=periodic,  # TODO: add periodic to proposals
        #update_fn=None,
        #update_iterations=-1,
        branch_names=["emri"],
        info={"truth":emri_injection_params_in}

    )

    # TODO: check about using injection as reference when the glitch is added
    # may need to add the heterodyning updater
    nsteps = 10000
    out = sampler.run_mcmc(start_state, nsteps, progress=True, thin_by=1, burn=0)

    # get samples
    samples = sampler.get_chain(discard=0, thin=1)["emri"][:, 0].reshape(-1, ndim)
    
    # plot
    fig = corner.corner(samples, levels=1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2))
    fig.savefig(fp[:-3] + "_corner.png", dpi=150)
    return

if __name__ == "__main__":
    # set parameters
    M = 1e6
    a = 0.1  # will be ignored in Schwarzschild waveform
    mu = 20.0
    p0 = 12.0
    e0 = 0.35
    x0 = 1.0  # will be ignored in Schwarzschild waveform
    qK = 0.2  # polar spin angle
    phiK = 0.2  # azimuthal viewing angle
    qS = 0.3  # polar sky angle
    phiS = 0.3  # azimuthal viewing angle
    dist = 3.0  # distance
    Phi_phi0 = 1.0
    Phi_theta0 = 2.0
    Phi_r0 = 3.0

    Tobs = 2.05
    dt = 15.0
    eps = 1e-5
    injectFD = 0
    fp = f"emri_M{M:.2}_mu{mu:.2}_p{p0:.2}_e{e0:.2}_T{Tobs}_eps{eps}_seed{SEED}_injectFD{injectFD}.h5"

    emri_injection_params = np.array([
        M,  
        mu, 
        a,
        p0, 
        e0, 
        x0, 
        dist, 
        qS, 
        phiS, 
        qK, 
        phiK, 
        Phi_phi0, 
        Phi_theta0, 
        Phi_r0
    ])

    ntemps = 2
    nwalkers = 32

    waveform_kwargs = {
        "T": Tobs,
        "dt": dt,
        "eps": eps
    }

    run_emri_pe(
        emri_injection_params, 
        Tobs,
        dt,
        fp,
        ntemps,
        nwalkers,
        emri_kwargs=waveform_kwargs
    )
