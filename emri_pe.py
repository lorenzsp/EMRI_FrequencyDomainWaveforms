# nohup python emri_pe.py > out.out &
# nohup python emri_pe.py -Tobs 2.0 -M 1e6 -mu 10.0 -p0 12.0 -e0 0.35 -dev 7 -eps 1e-3 -dt 10.0 -injectFD 1 -template fd -nwalkers 32 -ntemps 2 -downsample 1 > out4.out &
import argparse

# test on cpu
# python emri_pe.py -Tobs 4.0 -M 3670041.7362535275 -mu 292.0583167470244 -p0 13.709101864726545 -e0 0.5794130830706371 -eps 1e-2 -dt 10.0 -injectFD 1 -template fd -nwalkers 16 -ntemps 1 -downsample 1 -dev 0 -window_flag 0 
# python emri_pe.py -Tobs 4.0 -M 3670041.7362535275 -mu 292.0583167470244 -p0 13.709101864726545 -e0 0.5794130830706371 -eps 1e-2 -dt 10.0 -injectFD 1 -template fd -nwalkers 16 -ntemps 1 -downsample 0 -dev 0 -window_flag 0 

parser = argparse.ArgumentParser(description="MCMC few")
parser.add_argument(
    "-Tobs", "--Tobs", help="Observation Time in years", required=True, type=float
)
parser.add_argument(
    "-M", "--M", help="MBH Mass in solar masses", required=True, type=float
)
parser.add_argument(
    "-mu", "--mu", help="Compact Object Mass in solar masses", required=True, type=float
)
parser.add_argument("-p0", "--p0", help="Semi-latus Rectum", required=True, type=float)
parser.add_argument("-e0", "--e0", help="Eccentricity", required=True, type=float)
parser.add_argument("-dev", "--dev", help="Cuda Device", required=True, type=int)
parser.add_argument("-eps", "--eps", help="eps mode selection", required=True, type=float)
parser.add_argument("-dt", "--dt", help="delta t", required=True, type=float)
parser.add_argument("-injectFD", "--injectFD", required=True, type=int)
parser.add_argument("-template", "--template", required=True, type=str)
parser.add_argument("-downsample", "--downsample", required=True, type=int)
parser.add_argument("-nwalkers", "--nwalkers", required=True, type=int)
parser.add_argument("-ntemps", "--ntemps", required=True, type=int)
parser.add_argument("-window_flag", "--window_flag", required=True, type=int)

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

from eryn.moves import StretchMove, GaussianMove
from lisatools.sampling.likelihood import Likelihood
from lisatools.diagnostic import *
import multiprocessing as mp

# from lisatools.sensitivity import get_sensitivity
from FDutils import *
from scipy.signal.windows import (
    blackman,
    blackmanharris,
    hamming,
    hann,
    nuttall,
    parzen,
)

from few.waveform import GenerateEMRIWaveform
from few.utils.utility import get_p_at_t
from few.trajectory.inspiral import EMRIInspiral

from eryn.utils import TransformContainer
from few.utils.utility import omp_set_num_threads

omp_set_num_threads(1)

import time
import matplotlib.pyplot as plt
from few.utils.constants import *

SEED = 2601996
np.random.seed(SEED)

try:
    import cupy as xp

    # set GPU device
    xp.cuda.runtime.setDevice(args["dev"])
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
    return_list=False,
    # frame='source',
)


def transform_mass_ratio(logM, logeta):
    return [np.exp(logM), np.exp(logM) * np.exp(logeta)]


few_gen_list = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    use_gpu=use_gpu,
    return_list=True,
    # frame='source',
)

td_gen_list = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=dict(pad_output=True, odd_len=True),
    use_gpu=use_gpu,
    return_list=True,
    # frame='source',
)

td_gen = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux",
    sum_kwargs=dict(pad_output=True, odd_len=True),
    use_gpu=use_gpu,
    return_list=False,
    # frame='source',
)


# function call
def run_emri_pe(
    emri_injection_params,
    Tobs,
    dt,
    fp,
    ntemps,
    nwalkers,
    injectFD=1,
    template="fd",
    emri_kwargs={},
    downsample=False,
    window_flag=True,
):
    (
        M,
        mu,
        a,  # 2
        p0,
        e0,
        x0,  # 5
        dist,  # 6
        qS,
        phiS,
        qK,
        phiK,
        Phi_phi0,
        Phi_theta0,  # 12
        Phi_r0,
    ) = emri_injection_params

    # for transforms
    # this is an example of how you would fill parameters
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
        "ndim_full": 14,
        "fill_values": np.array(
            [0.0, x0, dist, qS, phiS, qK, phiK, Phi_theta0]
        ),  # spin and inclination and Phi_theta
        "fill_inds": np.array([2, 5, 6, 7, 8, 9, 10, 12]),
    }

    # mass ratio
    emri_injection_params[1] = np.log(
        emri_injection_params[1] / emri_injection_params[0]
    )
    # log of M mbh
    emri_injection_params[0] = np.log(emri_injection_params[0])

    # remove three we are not sampling from (need to change if you go to adding spin)
    emri_injection_params_in = np.delete(emri_injection_params, fill_dict["fill_inds"])

    # priors
    priors = {
        "emri": ProbDistContainer(
            {
                0: uniform_dist(np.log(5e5), np.log(1e7)),  # M
                1: uniform_dist(np.log(1e-6), np.log(1e-4)),  # mass ratio
                2: uniform_dist(10.0, 15.0),  # p0
                3: uniform_dist(0.001, 0.7),  # e0
                4: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                5: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            }
        )
    }

    # sampler treats periodic variables by wrapping them properly
    periodic = {"emri": {4: 2 * np.pi, 5: np.pi}}

    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
    parameter_transforms = {
        (0, 1): transform_mass_ratio,
    }

    transform_fn = TransformContainer(
        parameter_transforms=parameter_transforms,
        fill_dict=fill_dict,
    )

    # get injected parameters after transformation
    injection_in = transform_fn.both_transforms(emri_injection_params_in[None, :])[0]

    # generate FD waveforms
    data_channels_fd = few_gen(*injection_in, **emri_kwargs)
    
    # timing
    # tic = time.perf_counter()
    # [few_gen(*injection_in, **emri_kwargs) for _ in range(3)]
    # toc = time.perf_counter()
    # fd_time = toc-tic
    # print('fd time', fd_time/3)
    # frequency goes from -1/dt/2 up to 1/dt/2
    frequency = few_gen.waveform_generator.create_waveform.frequency
    positive_frequency_mask = frequency >= 0.0
    # transform into hp and hc
    emri_kwargs["mask_positive"] = True
    sig_fd = few_gen_list(*injection_in, **emri_kwargs)
    del emri_kwargs["mask_positive"]
    # non zero frequencies
    non_zero_mask = xp.abs(sig_fd[0]) > 1e-50

    # generate TD waveform, this will return a list with hp and hc
    data_channels_td = td_gen_list(*injection_in, **emri_kwargs)
    
    # timing
    # tic = time.perf_counter()
    # [td_gen(*injection_in, **emri_kwargs) for _ in range(3)]
    # toc = time.perf_counter()
    # fd_time = toc-tic
    # print('td time', fd_time/3)
    
    # windowing signals
    if window_flag:
        window = xp.asarray(hann(len(data_channels_td[0])))
        fft_td_gen = get_fd_waveform_fromTD(td_gen_list, positive_frequency_mask, dt, window=window)
        fd_gen = get_fd_waveform_fromFD(few_gen_list, positive_frequency_mask, dt, window=window)
    else:
        window = None
        fft_td_gen = get_fd_waveform_fromTD(td_gen_list, positive_frequency_mask, dt, window=window)
        fd_gen = get_fd_waveform_fromFD(few_gen_list, positive_frequency_mask, dt, window=window)

    # injections
    sig_fd = fd_gen(*injection_in, **emri_kwargs)
    sig_td = fft_td_gen(*injection_in, **emri_kwargs)

    # kwargs for computing inner products
    print("shape", sig_td[0].shape, sig_fd[0].shape)
    if use_gpu:
        fd_inner_product_kwargs = dict(
            PSD=xp.asarray(get_sensitivity(frequency[positive_frequency_mask].get())),
            use_gpu=use_gpu,
            f_arr=frequency[positive_frequency_mask],
        )
    else:
        fd_inner_product_kwargs = dict(
            PSD=xp.asarray(get_sensitivity(frequency[positive_frequency_mask])),
            use_gpu=use_gpu,
            f_arr=frequency[positive_frequency_mask],
        )

    print(
        "Overlap total and partial ",
        inner_product(sig_fd, sig_td, normalize=True, **fd_inner_product_kwargs),
        inner_product(sig_fd[0], sig_td[0], normalize=True, **fd_inner_product_kwargs),
        inner_product(sig_fd[1], sig_td[1], normalize=True, **fd_inner_product_kwargs),
    )

    print("frequency len", len(frequency), " make sure that it is odd")
    print("last point in TD", data_channels_td[0][-1])
    check_snr = snr(sig_fd, **fd_inner_product_kwargs)
    print("SNR = ", check_snr)

    # this is a parent likelihood class that manages the parameter transforms
    nchannels = 2
    if template == "fd":
        like_gen = fd_gen
    elif template == "td":
        like_gen = fft_td_gen

    # inject a signal
    if bool(injectFD):
        data_stream = sig_fd
    else:
        data_stream = sig_td

    if use_gpu:
        plt.figure()
        plt.loglog(np.abs(data_stream[0].get()) ** 2)
        plt.savefig(fp[:-3] + "injection.pdf")
    else:
        plt.figure()
        plt.loglog(np.abs(data_stream[0]) ** 2)
        plt.savefig(fp[:-3] + "injection.pdf")

    if downsample:
        if template == "td":
            raise ValueError("Cannot run downsampling with time domain template")
        else:
            print("Running with downsampling, injecing consistently the FD signal")
        # downsample the fft of the window
        if window_flag:
            raise ValueError("Cannot run downsampling with windowing")

        # list the indeces
        lst_ind = list(range(len(frequency)))
        upp = 30

        
        # make sure there is the zero frequency when you jump
        check_vec = xp.asarray([1 == xp.sum(frequency[lst_ind[0::ii]] == 0.0) for ii in range(2, upp)])
        # find the one that has the zero frequency
        ii = int(xp.arange(2,upp)[check_vec][-1])
        fp += f'_skip{ii}'
        print('--------------------------')
        print('skip every ',ii, 'th element')
        print('number of frequencies', len(frequency[lst_ind[0::ii]]))
        print('percentage of frequencies', len(frequency[lst_ind[0::ii]])/len(frequency))
        # add f_arr to the kwarguments
        newfreq = frequency[lst_ind[0::ii]]
        # newfreq = xp.hstack((-10.0**xp.linspace(-5,-1,num=1000),0.0,10.0**xp.linspace(-5,-1,num=1000)) )

        emri_kwargs["f_arr"] = newfreq
        if use_gpu:
            # get the index of the positive frequencies
            f_arr = newfreq[newfreq >= 0.0].get()
        else:
            # get the index of the positive frequencies
            f_arr = newfreq[newfreq >= 0.0]
        
        # take full array of frequencies and set it to true
        all_freq_true = xp.ones_like(lst_ind,dtype=bool)
        # set to negative the others
        all_freq_true[lst_ind[0::ii]] *= False
        all_freq_true = ~all_freq_true
        ind = all_freq_true[positive_frequency_mask]

        # modify the positive frequencies with the downsamples version
        positive_frequency_mask = (newfreq >= 0.0)
        # define the new waveform generator for the likelihood
        like_gen = get_fd_waveform_fromFD(few_gen_list, positive_frequency_mask, dt, window=window)
        # define the kwargs for the innerproduct
        fd_inner_product_kwargs_downsamp = dict(PSD=xp.asarray(get_sensitivity(f_arr)), use_gpu=use_gpu, f_arr=f_arr)
        
        # make the check of the downsamples data stream
        check_downsampled = like_gen(*injection_in, **emri_kwargs)
        # timing
        tic = time.perf_counter()
        [like_gen(*injection_in, **emri_kwargs) for _ in range(3)]
        toc = time.perf_counter()
        fd_time = toc-tic
        print('fd time', fd_time/3)
        # take the previous datastream and downsample
        data_stream = [el[ind] for el in sig_fd]
        print("Overlap = ",inner_product(data_stream, check_downsampled, **fd_inner_product_kwargs_downsamp, normalize=True))
        print("SNR = ", snr(data_stream, **fd_inner_product_kwargs_downsamp))
        print("SNR = ", snr(check_downsampled, **fd_inner_product_kwargs_downsamp))
        # plt.figure(); plt.loglog(np.abs(data_stream[0]) ** 2);plt.loglog(np.abs(check_downsampled[0]) ** 2, '--');plt.savefig("test_data.pdf")

    else:
        if use_gpu:
            f_arr = frequency[positive_frequency_mask].get()
        else:
            f_arr = frequency[positive_frequency_mask]

    # if use_gpu:
    like = Likelihood(
        like_gen,
        nchannels,  # channels (plus,cross)
        parameter_transforms={"emri": transform_fn},
        vectorized=False,
        transpose_params=False,
        subset=24,  # may need this subset
        f_arr=f_arr,
        use_gpu=use_gpu,
    )

    like.inject_signal(
        data_stream=data_stream,
        # params= injection_params.copy()[test_inds],
        waveform_kwargs=emri_kwargs,
        noise_fn=[get_sensitivity, get_sensitivity],
        noise_args=[(), ()],
        noise_kwargs=[{}, {}],  # dict(sens_fn="cornish_lisa_psd"),
        add_noise=False,
    )
    # gpu samples
    gpusamp = np.load("samples_GPU.npy")
    for ii in range(10):
        print( like(gpusamp[ii,:-1], **emri_kwargs)-gpusamp[ii,-1] )
    breakpoint()

    # dimensions of the sampling parameter space
    ndim = 6

    # generate starting points
    factor = 1e-5
    cov = np.cov(np.load("covariance.npy"), rowvar=False) / (2.4 * ndim)

    start_params = np.random.multivariate_normal(
        emri_injection_params_in, cov, size=nwalkers * ntemps
    )
    start_prior = priors["emri"].logpdf(start_params)
    start_like = like(start_params, **emri_kwargs)
    start_params[np.isnan(start_like)] = np.random.multivariate_normal(
        emri_injection_params_in, cov, size=start_params[np.isnan(start_like)].size
    )
    print("likelihood", start_like)
    print("likelihood injection", like(emri_injection_params_in[:, None].T, **emri_kwargs))

    # start state
    start_state = State(
        {"emri": start_params.reshape(ntemps, nwalkers, 1, ndim)},
        log_like=start_like.reshape(ntemps, nwalkers),
        log_prior=start_prior.reshape(ntemps, nwalkers),
    )

    # MCMC gibbs
    update_all = np.repeat(True, ndim)
    update_none = np.repeat(False, ndim)
    indx_list = []

    def get_True_vec(ind_in):
        out = update_none.copy()
        out[ind_in] = update_all[ind_in]
        return out

    # gibbs sampling setup
    indx_list.append(get_True_vec(np.arange(0, 4)))
    indx_list.append(get_True_vec(np.arange(4, ndim)))
    gibbs_sampling = [
        ("emri", np.asarray([indx_list[ii]])) for ii in range(len(indx_list))
    ]

    # define move
    moves = [
        # GaussianMove({"emri": cov}, factor=1000, gibbs_sampling_setup=gibbs_sampling)
        StretchMove(
            use_gpu=use_gpu, live_dangerously=True
        )  # , gibbs_sampling_setup=gibbs_sampling)
    ]

    # define stopping function
    start = time.time()

    def get_time(i, res, samp):
        if i % 50 == 0:
            print("acceptance ratio", samp.acceptance_fraction)
            print("max last loglike", np.max(samp.get_log_like()[-1]))
            # if (i>100)and(i<1000):
            #     emrisamp = samp.get_chain()['emri'][-100:,0][samp.get_inds()['emri'][-100:,0]]
            #     samp.moves[0].all_proposal['emri'].scale = np.cov(emrisamp,rowvar=False)

        # if time.time()-start > 23.0*3600:
        #     return True
        # else:
        return False

    from eryn.backends import HDFBackend

    # check for previous runs
    try:
        file_samp = HDFBackend(fp)
        last_state = file_samp.get_last_sample()
        inds = last_state.branches_inds.copy()
        new_coords = last_state.branches_coords.copy()
        coords = new_coords.copy()
        resume = True
        print("resuming")
    except:
        resume = False
        print("file not found")
    import pickle

    nsteps = 500_000

    if use_gpu:
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
            periodic=periodic,
            # update_fn=None,
            # update_iterations=-1,
            stopping_fn=get_time,
            stopping_iterations=1,
            branch_names=["emri"],
            info={"truth": emri_injection_params_in},
        )

        if resume:
            log_prior = sampler.compute_log_prior(coords, inds=inds)
            log_like = sampler.compute_log_like(coords, inds=inds, logp=log_prior)[0]
            print("initial loglike", log_like)
            start_state = State(coords, log_like=log_like, log_prior=log_prior, inds=inds)

        out = sampler.run_mcmc(start_state, nsteps, progress=True, thin_by=1, burn=0)

    else:
        # use multiprocessing only on CPUs
        with mp.Pool(16) as pool:
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
                vectorize=False,
                pool=pool,
                periodic=periodic,
                # update_fn=None,
                # update_iterations=-1,
                stopping_fn=get_time,
                stopping_iterations=1,
                branch_names=["emri"],
                info={"truth": emri_injection_params_in},
            )

            if resume:
                log_prior = sampler.compute_log_prior(coords, inds=inds)
                log_like = sampler.compute_log_like(coords, inds=inds, logp=log_prior)[0]
                print("initial loglike", log_like)
                start_state = State(
                    coords, log_like=log_like, log_prior=log_prior, inds=inds
                )

            out = sampler.run_mcmc(start_state, nsteps, progress=True, thin_by=1, burn=0)

    # get samples
    samples = sampler.get_chain(discard=0, thin=1)["emri"][:, 0].reshape(-1, ndim)

    # plot
    fig = corner.corner(samples, levels=1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2))
    fig.savefig(fp[:-3] + "_corner.png", dpi=150)
    return


if __name__ == "__main__":

    window_flag = bool(args["window_flag"])
    downsample = bool(args["downsample"])
    Tobs = args["Tobs"]  # years
    dt = args["dt"]  # seconds
    eps = args["eps"]  # threshold mode content
    injectFD = args["injectFD"]  # 0 = inject TD
    template = args["template"]  #'fd'

    # set parameters
    M = args["M"]  # 1e6
    a = 0.1  # will be ignored in Schwarzschild waveform
    mu = args["mu"]  # 10.0
    p0 = args["p0"]  # 12.0
    e0 = args["e0"]  # 0.35
    x0 = 1.0  # will be ignored in Schwarzschild waveform
    qK = np.pi / 3  # polar spin angle
    phiK = np.pi / 3  # azimuthal viewing angle
    qS = np.pi / 3  # polar sky angle
    phiS = np.pi / 3  # azimuthal viewing angle
    if window_flag:
        dist = 1
    else:
        dist = 2.4539054256
    Phi_phi0 = np.pi / 3
    Phi_theta0 = 0.0
    Phi_r0 = np.pi / 3

    ntemps = args["ntemps"]
    nwalkers = args["nwalkers"]

    traj = EMRIInspiral(func="SchwarzEccFlux")

    # fix p0 given T
    p0 = get_p_at_t(
        traj,
        Tobs * 0.99,
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

    # name output
    fp = f"results/MCMC_M{M:.2}_mu{mu:.2}_p{p0:.2}_e{e0:.2}_T{Tobs}_eps{eps}_seed{SEED}_nw{nwalkers}_nt{ntemps}_downsample{int(downsample)}_injectFD{injectFD}_usegpu{str(use_gpu)}_template{template}_window_flag{window_flag}.h5"

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
        downsample=downsample,
        injectFD=injectFD,
        window_flag=window_flag,
    )
