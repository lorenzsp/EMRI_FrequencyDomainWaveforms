import argparse
# python downsampled_pe.py -Tobs 2.0 -M 1e6 -mu 10.0 -p0 12 -e0 0.35 -dev 5 -eps 1e-3 -dt 10.0 -injectFD 1 -template fd -nwalkers 128 -ntemps 1 -downsample 0
parser = argparse.ArgumentParser(description='MCMC few')
parser.add_argument('-Tobs','--Tobs', help='Observation Time in years', required=True, type=float)
parser.add_argument('-M','--M', help='MBH Mass in solar masses', required=True, type=float)
parser.add_argument('-mu','--mu', help='Compact Object Mass in solar masses', required=True, type=float)
parser.add_argument('-p0','--p0', help='Semi-latus Rectum', required=True, type=float)
parser.add_argument('-e0','--e0', help='Eccentricity', required=True, type=float)
parser.add_argument('-dev','--dev', help='Cuda Device', required=True, type=int)
parser.add_argument('-eps','--eps', help='eps mode selection', required=True, type=float)
parser.add_argument('-dt','--dt', help='delta t', required=True, type=float)
parser.add_argument('-injectFD','--injectFD', required=True, type=int)
parser.add_argument('-template','--template', required=True, type=str)
parser.add_argument('-downsample','--downsample', required=True, type=int)
parser.add_argument('-nwalkers','--nwalkers', required=True, type=int)
parser.add_argument('-ntemps','--ntemps', required=True, type=int)

args = vars(parser.parse_args())


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
from few.utils.utility import get_p_at_t
from few.trajectory.inspiral import EMRIInspiral

from eryn.utils import TransformContainer
from few.utils.utility import omp_set_num_threads
omp_set_num_threads(16)

import time
import matplotlib.pyplot as plt
from few.utils.constants import *
SEED=2601996
np.random.seed(SEED)

try:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(args['dev'])
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
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    use_gpu=use_gpu,
    return_list=False
)

few_gen_list = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux", 
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    use_gpu=use_gpu,
    return_list=True
)

td_gen_list = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux", 
    sum_kwargs=dict(pad_output=True, odd_len=True),
    use_gpu=use_gpu,
    return_list=True
)

td_gen = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux", 
    sum_kwargs=dict(pad_output=True, odd_len=True),
    use_gpu=use_gpu,
    return_list=False
)


# conversion
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
    downsample = False,
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
    emri_injection_params[1] = np.log(emri_injection_params[1]/emri_injection_params[0])
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
                1: uniform_dist(np.log(1e-6), np.log(1e-4)),  # mass ratio
                2: uniform_dist(8.0, 15.0),  # p0
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

    def transform_mass_ratio(logM, logeta):
        return [np.exp(logM),  np.exp(logM) * np.exp(logeta)]

    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
    parameter_transforms = {
        (0,1): transform_mass_ratio,
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
    tic = time.perf_counter()
    [few_gen(*injection_in, **emri_kwargs) for _ in range(10)]
    toc = time.perf_counter()
    fd_time = toc-tic
    print('fd time', fd_time/10)
    # frequency goes from -1/dt/2 up to 1/dt/2
    frequency = few_gen.waveform_generator.create_waveform.frequency
    positive_frequency_mask = (frequency>=0.0)
    # transform into hp and hc
    emri_kwargs['mask_positive']=True
    sig_fd = few_gen_list(*injection_in, **emri_kwargs)
    
    # generate TD waveform, this will return a list with hp and hc
    data_channels_td = td_gen_list(*injection_in, **emri_kwargs)
    tic = time.perf_counter()
    [td_gen(*injection_in, **emri_kwargs) for _ in range(10)]
    toc = time.perf_counter()
    fd_time = toc-tic
    print('td time', fd_time/10)
    # fft from negative to positive frequencies
    fft_td_wave_c = xp.fft.fftshift(xp.fft.fft(data_channels_td[1])) * dt
    fft_td_wave_p = xp.fft.fftshift(xp.fft.fft(data_channels_td[0])) * dt
    # consider only positive frequencies
    fft_td_gen = get_fd_waveform_fromTD(td_gen_list, positive_frequency_mask, dt)
    print('check TD transform', xp.all(fft_td_gen(*injection_in, **emri_kwargs)[0] == fft_td_wave_p[positive_frequency_mask]) )
    sig_td = [fft_td_wave_p[positive_frequency_mask],fft_td_wave_c[positive_frequency_mask] ]

    # kwargs for computing inner products
    print('shape', sig_td[0].shape, sig_fd[0].shape )
    fd_inner_product_kwargs = dict( PSD="cornish_lisa_psd", use_gpu=use_gpu, f_arr=frequency[positive_frequency_mask])

    print("Overlap total and partial ", inner_product(sig_fd, sig_td, normalize=True, **fd_inner_product_kwargs),
    inner_product(sig_fd[0], sig_td[0], normalize=True, **fd_inner_product_kwargs),
    inner_product(sig_fd[1], sig_td[1], normalize=True, **fd_inner_product_kwargs)
    )
    
    print("frequency len",len(frequency), " make sure that it is odd")
    print("last point in TD", data_channels_td[0][-1])
    check_snr = snr(sig_fd, **fd_inner_product_kwargs)
    print("SNR = ", check_snr)

    # this is a parent likelihood class that manages the parameter transforms
    nchannels = 2
    if template=='fd':
        like_gen = few_gen_list
    elif template=='td':
        like_gen = fft_td_gen
    
    # inject a signal
    if bool(injectFD):
        data_stream = sig_fd
    else:
        data_stream = sig_td

    # ---------------------------------------------
    # do the standard likelihood
    if use_gpu:
        f_arr = frequency[positive_frequency_mask].get()
    else:
        f_arr = frequency[positive_frequency_mask]

    like = Likelihood(
        like_gen,
        nchannels,  # channels (plus,cross)
        parameter_transforms={"emri": transform_fn},
        vectorized=False,
        transpose_params=False,
        subset=8,  # may need this subset
        f_arr = f_arr,
        use_gpu=use_gpu
    )

    like.inject_signal(
        data_stream=data_stream,
        # params= injection_params.copy()[test_inds],
        waveform_kwargs=emri_kwargs,
        noise_fn=get_sensitivity,
        noise_kwargs=dict(sens_fn="cornish_lisa_psd"),
        add_noise=False,
    )

    # ---------------------------------------------
    # do the downsampled likelihood
    # list the indeces 
    lst_ind = list(range(len(frequency)))
    # make sure there is the zero frequency when you jump
    upper = 10_000
    check_vec = xp.asarray([1==xp.sum(frequency[lst_ind[0::ii]]==0.0) for ii in range(2,upper)])
    # find the one that has the zero frequency
    ii = int(xp.arange(2,upper)[check_vec][-1])
    print('--------------------------')
    print('skip every ',ii, 'th element')
    print('number of frequencies', len(frequency[lst_ind[0::ii]]))
    print('percentage of frequencies', len(frequency[lst_ind[0::ii]])/len(frequency))
    # add f_arr to the kwarguments
    ds_kw = emri_kwargs.copy()
    ds_kw['f_arr'] = frequency[lst_ind[0::ii]]
    if use_gpu:
        f_arr = frequency[lst_ind[0::ii]][frequency[lst_ind[0::ii]]>=0.0].get()
    else:
        f_arr = frequency[lst_ind[0::ii]][frequency[lst_ind[0::ii]]>=0.0]
    # downsample data stream
    data_stream = [el[0::ii] for el in data_stream]

    like_downsampled = Likelihood(
        few_gen_list,
        nchannels,  # channels (plus,cross)
        parameter_transforms={"emri": transform_fn},
        vectorized=False,
        transpose_params=False,
        subset=8,  # may need this subset
        f_arr = f_arr,
        use_gpu=use_gpu
    )

    like_downsampled.inject_signal(
        data_stream=data_stream,
        # params= injection_params.copy()[test_inds],
        waveform_kwargs=ds_kw,
        noise_fn=get_sensitivity,
        noise_kwargs=dict(sens_fn="cornish_lisa_psd"),
        add_noise=False,
    )

    ###############################################
    ndim = 11

    # generate starting points
    factor = 1e-15
    
    cov = factor * np.load("covariance.npy")
    start_params = np.random.multivariate_normal(emri_injection_params_in, cov, size=nwalkers * ntemps)
    start_params = np.load("samples_to_test.npy")[:int(nwalkers * ntemps)]

    start_prior = priors["emri"].logpdf(start_params)
    
    start_like = []
    tic = time.perf_counter()
    start_like.append(like(start_params, **emri_kwargs))
    toc = time.perf_counter()
    print("timing", (toc - tic)/(nwalkers * ntemps) )

    tic = time.perf_counter()
    start_like.append(like_downsampled(start_params, **ds_kw)) 
    toc = time.perf_counter()
    print("timing", (toc - tic)/(nwalkers * ntemps) )

    start_like = np.asarray(start_like)
    like_diff = np.diff(start_like,axis=0).flatten()
    plt.figure(); plt.plot(start_like[0], like_diff,'.'); plt.savefig('likediff.png')

    print("likelihood",start_like)

    return

if __name__ == "__main__":
    
    # set parameters
    M = args['M'] # 1e6
    a = 0.1  # will be ignored in Schwarzschild waveform
    mu = args['mu']# 10.0
    p0 = args['p0'] # 12.0
    e0 = args['e0'] # 0.35
    x0 = 1.0  # will be ignored in Schwarzschild waveform
    qK = np.pi/3  # polar spin angle
    phiK = np.pi/4  # azimuthal viewing angle
    qS = np.pi/3  # polar sky angle
    phiS = np.pi/4  # azimuthal viewing angle
    dist = 3.0  # distance
    Phi_phi0 = 1.0
    Phi_theta0 = 2.0
    Phi_r0 = 3.0

    Tobs = args['Tobs'] # 1.00
    dt = args['dt'] # 1.0 # 4 Hz is the baseline 
    eps = args['eps'] # 1e-5
    injectFD = args['injectFD'] #0
    template = args['template'] #'fd'
    downsample = bool(args['downsample'])

    ntemps = args['ntemps']
    nwalkers = args['nwalkers']

    traj = EMRIInspiral(func="SchwarzEccFlux")

    p0 = get_p_at_t(
    traj,
    Tobs * 0.9,
    [M, mu, 0.0, e0, 1.0],
    index_of_p=3,
    index_of_a=2,
    index_of_e=4,
    index_of_x=5,
    traj_kwargs={},
    xtol=2e-12,
    rtol=8.881784197001252e-16,
    bounds=None,
    )
    print("new p0 ", p0)

    
    fp = f"emri_M{M:.2}_mu{mu:.2}_p{p0:.2}_e{e0:.2}_T{Tobs}_eps{eps}_seed{SEED}_nw{nwalkers}_nt{ntemps}_downsample{int(downsample)}_injectFD{injectFD}_template" + template + ".h5"

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
        emri_kwargs=waveform_kwargs,
        template=template,
        downsample=downsample
    )
