import os
print("PID:",os.getpid())
# from few.utils.utility import omp_set_num_threads

# os.system("export OMP_NUM_THREADS=4")
# os.environ["OMP_NUM_THREADS"] = "4"

import argparse
# python check_mode_by_mode.py -Tobs 1.0 -dev 5 -eps 1e-3 -dt 10.0 -fixed_insp 1
parser = argparse.ArgumentParser(description='MCMC few')
parser.add_argument('-Tobs','--Tobs', help='Observation Time in years', required=True, type=float)
parser.add_argument('-dev','--dev', help='Cuda Device', required=True, type=int)
parser.add_argument('-eps','--eps', help='eps mode selection', required=True, type=float)
parser.add_argument('-dt','--dt', help='delta t', required=False, type=float, default=10.0)
parser.add_argument('-fixed_insp','--fixed_insp', help='fix mu to get inspiral Tobs', required=False, type=int, default=1)

args = vars(parser.parse_args())

import sys
sys.path.append("../LISAanalysistools/")
sys.path.append("../Eryn/")

import numpy as np
from eryn.state import State
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
import corner

from eryn.moves import StretchMove
from lisatools.sampling.likelihood import Likelihood
from lisatools.diagnostic import *

# from lisatools.sensitivity import get_sensitivity
from FDutils import *

from few.waveform import GenerateEMRIWaveform
from few.utils.utility import get_mu_at_t, get_p_at_t
from few.trajectory.inspiral import EMRIInspiral
traj_module = EMRIInspiral(func="SchwarzEccFlux")

from eryn.utils import TransformContainer
from few.utils.baseclasses import SchwarzschildEccentric
from few.utils.utility import omp_set_num_threads

import time

import h5py
from scipy.signal.windows import blackman, blackmanharris, hamming, hann, nuttall, parzen

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

frame = 'detector'
few_gen = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux", 
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    use_gpu=use_gpu,
    return_list=False,
    # frame=frame,
)

few_gen_list = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux", 
    sum_kwargs=dict(pad_output=True, output_type="fd", odd_len=True),
    use_gpu=use_gpu,
    return_list=True,
    # frame=frame,
)

td_gen_list = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux", 
    sum_kwargs=dict(pad_output=True, odd_len=True),
    use_gpu=use_gpu,
    return_list=True,
    # frame=frame,
)

td_gen = GenerateEMRIWaveform(
    "FastSchwarzschildEccentricFlux", 
    sum_kwargs=dict(pad_output=True, odd_len=True),
    use_gpu=use_gpu,
    return_list=False,
    # frame=frame,
)

# function call
def run_check(
    Tobs,
    dt,
    fp,
    injectFD=1,
    template='fd',
    emri_kwargs={},
    random_modes=False,
    get_fixed_inspiral=True,
    tot_numb = 50,
):

    # for transforms
    # this is an example of how you would fill parameters 
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict = {
       "ndim_full": 14,
       "fill_values": np.array([0.0, 1.0, 1.0, np.pi/3, np.pi/3, np.pi/3, np.pi/3, 0.0]), # spin and inclination and Phi_theta
       "fill_inds":   np.array([2,   5,  6,     7,  8,   9,   10,    12]),
    }

    # priors
    priors = {
        "emri": ProbDistContainer(
            {
                0: uniform_dist(np.log(1e5), np.log(1e7)),  # M
                1: uniform_dist(np.log(1e-6), np.log(1e-4)),  # mass ratio
                2: uniform_dist(10.0, 16.0),  # p0
                3: uniform_dist(0.001, 0.7),  # e0
                4: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
                5: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            }
        ) 
    }

    # sampler treats periodic variables by wrapping them properly
    periodic = {
        "emri": {4: 2 * np.pi, 5: 2 * np.pi}
    }

    def transform_mass_ratio(logM, logeta):
        return [np.exp(logM),  np.exp(logM) * np.exp(logeta)]

    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
    parameter_transforms = {
        (0,1): transform_mass_ratio,
    }

    transform_fn = TransformContainer(
        parameter_transforms=parameter_transforms,
        fill_dict=fill_dict,
    )

    dset = h5py.File(fp + '.h5',mode='w')
    factor = []
    mismatch = []
    failed_points = []
    list_injections = []
    timing_td = []
    timing_fd = []
    loglike = []
    SNR_list = []
    
    for el in range(tot_numb):
        
        print( el/tot_numb,'---------------------')
        if random_modes:
            try:
                del emri_kwargs['eps']
            except:
                pass
            ll = np.random.randint(2,10)
            mm = np.random.randint(-ll,ll+1)
            nn = np.random.randint(-30,30)
            print(ll, mm, nn)
            emri_kwargs['mode_selection'] = [(ll, mm, nn)]

        # randomly draw parameters
        tmp = priors["emri"].rvs()
        # get injected parameters after transformation
        injection_in = transform_fn.both_transforms(tmp)[0]

        # set initial parameters
        M = injection_in[0]
        mu = injection_in[1]
        p0 = injection_in[3]
        e0 = injection_in[4]

        # get p in order to get an inspiral of 
        t_out = Tobs*0.99
        
        try:
            # run trajectory to get fixed inspiral
            if get_fixed_inspiral:
                print('params ', M, mu, p0, e0)
                p0 = get_p_at_t(
                traj_module,
                t_out,
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
                injection_in[3] = p0
                # mu = get_mu_at_t(traj_module,t_out,[M, 0.0, p0, e0, 1.0],index_of_mu=1,traj_kwargs={},xtol=2e-6,rtol=8.881784197001252e-6,bounds=[0.1,1e4])
                # injection_in[1] = mu
            print('params ', M, mu, p0, e0)
            
            check = SchwarzschildEccentric()
            check.sanity_check_init(M,mu,p0,e0)

            it_speed = 3
            #-------------------------
            tic = time.perf_counter()
            # generate FD waveforms
            for _ in range(it_speed):
                few_gen(*injection_in, **emri_kwargs)
            # transform into hp and hc
            toc = time.perf_counter()
            fd_time = (toc-tic)/it_speed
            data_channels_fd = few_gen(*injection_in, **emri_kwargs)

            # downsampled
            kw_downsampled = emri_kwargs.copy()
            kw_downsampled['f_arr'] = xp.fft.fftshift(xp.fft.fftfreq( int(len(data_channels_fd)*0.01) , dt))
            tic = time.perf_counter()
            # generate FD waveforms
            for _ in range(it_speed):
                few_gen(*injection_in, **kw_downsampled)
            # transform into hp and hc
            toc = time.perf_counter()
            fd_time_downsampled = (toc-tic)/it_speed

            
            #-------------------------
            # list version
            sig_fd = few_gen_list(*injection_in, **emri_kwargs)
            print("check 1 == ", xp.dot(xp.conj(sig_fd[0] - 1j * sig_fd[1]),data_channels_fd)/xp.dot(xp.conj(data_channels_fd),data_channels_fd) )

            # frequency goes from -1/dt/2 up to 1/dt/2
            frequency = few_gen_list.waveform_generator.create_waveform.frequency
            positive_frequency_mask = (frequency>=0.0)
            mask_non_zero = (sig_fd[0][positive_frequency_mask]!=complex(0.0))
            #-------------------------
            tic = time.perf_counter()
            # generate TD waveform, this will return a list with hp and hc
            for _ in range(it_speed):
                td_gen(*injection_in, **emri_kwargs)
            toc = time.perf_counter()
            td_time = (toc-tic)/it_speed
            data_channels_td = td_gen(*injection_in, **emri_kwargs)
            #-------------------------
            # list version
            signal_in_td = td_gen_list(*injection_in, **emri_kwargs)
            sig_td = get_fft_td_windowed(signal_in_td, 1.0, dt)
            
            # windowed verions
            # window = xp.asarray(tukey(len(data_channels_td), alpha=0.1))
            # window = xp.asarray(hann(len(data_channels_td)))
            sig_fd_windowed = [[el[positive_frequency_mask] for el in get_fd_windowed(sig_fd, xp.asarray(ww(len(data_channels_td))) )] 
                                for ww in [blackman, blackmanharris, hamming, hann, nuttall, parzen]]
            sig_td_windowed = [[el[positive_frequency_mask] for el in get_fft_td_windowed(signal_in_td, xp.asarray(ww(len(data_channels_td))), dt)]
                                for ww in [blackman, blackmanharris, hamming, hann, nuttall, parzen]]

            # store timing
            timing_td.append(td_time)
            timing_fd.append([fd_time, fd_time_downsampled])
            list_injections.append(injection_in)
            print("TD/FD time",td_time/fd_time, "TD", td_time, "FD", fd_time, "FD downsampled", fd_time_downsampled )
            factor.append(td_time/fd_time)

            # kwargs for computing inner products
            fd_inner_product_kwargs = dict( PSD="cornish_lisa_psd", use_gpu=use_gpu, f_arr=frequency[positive_frequency_mask])

            sig_fd = [el[positive_frequency_mask] for el in sig_fd]
            sig_td = [el[positive_frequency_mask] for el in sig_td]

            # get SNR
            SNR = [np.sqrt(float(inner_product(el, el, **fd_inner_product_kwargs))) for el in [sig_fd]+sig_fd_windowed]
            print('SNR', SNR)
            SNR_list.append(SNR)
            # norm = 20.0/SNR

            # mismatch 
            Mism = xp.abs(1-inner_product(sig_fd, sig_td, normalize=True, **fd_inner_product_kwargs)).get() 
            Mism_wind = [xp.abs(1-inner_product(el_fd, el_td, normalize=True, **fd_inner_product_kwargs)).get() 
                        for el_fd, el_td in zip(sig_fd_windowed, sig_td_windowed)]

            print("mismatch", Mism, Mism_wind)

            if use_gpu:
                mismatch.append([Mism]+Mism_wind)
            else:
                mismatch.append([Mism, Mism_wind])
            
            # loglike
            sig_inner = [sig_fd[0]-sig_td[0],sig_fd[1]-sig_td[1]]
            if use_gpu:
                logl = -0.5 * sum([inner_product(el, el, normalize=False, **fd_inner_product_kwargs).get() for el in sig_inner])
                logl_windowed = [-0.5 * sum([inner_product([el_fd[0]-el_td[0]], [el_fd[0]-el_td[0]], normalize=False, **fd_inner_product_kwargs).get()+
                                            inner_product([el_fd[1]-el_td[1]], [el_fd[1]-el_td[1]], normalize=False, **fd_inner_product_kwargs).get()])
                                for el_fd, el_td in zip(sig_fd_windowed, sig_td_windowed)]
            else:
                logl = -0.5 * sum([inner_product(el, el, normalize=False, **fd_inner_product_kwargs) for el in sig_inner])
                logl_windowed = -0.5 * sum([inner_product(el, el, normalize=False, **fd_inner_product_kwargs) for el in sig_inner_windowed])

            # if logl<-10.0:
            #     toplot = (sig_fd[0]-sig_td[0])[mask_non_zero]
            #     ff = frequency[positive_frequency_mask][mask_non_zero]
            #     if random_modes:
            #         mode = emri_kwargs['mode_selection'][0]
            #     else:
            #         mode = emri_kwargs['eps']

            #     if use_gpu:
            #         plt.figure(); plt.title(f'mismatch = {Mism}'); plt.loglog(ff.get(), xp.abs(toplot).get()**2 ); plt.savefig(f'high_mism/logl{logl}_{mode}_{M, mu, p0, e0}.png')
            #         plt.figure(); plt.loglog(ff.get(), xp.abs(sig_fd[0][mask_non_zero]).get()**2 ,label='FD'); plt.loglog(ff.get(), xp.abs(sig_td[0][mask_non_zero]).get()**2 ,'--',label='TD'); plt.legend(); plt.savefig(f'high_mism/mism{Mism}_{mode}_{M, mu, p0, e0}.png')
            #     else:
            #         plt.figure(); plt.loglog(ff, xp.abs(toplot)**2 ); plt.savefig(f'high_mism/logl{logl}.png')

            print("logl ", logl, logl_windowed)
            loglike.append([logl] + logl_windowed)

        except:
            # breakpoint()
            failed_points.append(injection_in)
            print("not found for params",tmp[:3])
    
    # store to h5 file
    dset.create_dataset("T", data=emri_kwargs['T'] )
    dset.create_dataset("dt", data=emri_kwargs['dt'] )
    
    if random_modes==False:
        dset.create_dataset("eps", data=emri_kwargs['eps'] )
    
    to_store = [
    mismatch,
    failed_points,
    list_injections,
    timing_td,
    timing_fd,
    loglike,
    SNR_list
    ]

    for el in to_store:
        el = np.asarray(el)
    
    dset.create_dataset("mismatch", data=mismatch)
    dset.create_dataset("failed_points", data=failed_points)
    dset.create_dataset("list_injections", data=list_injections)
    dset.create_dataset("timing_td", data=timing_td)
    dset.create_dataset("timing_fd", data=timing_fd)
    dset.create_dataset("loglike", data=loglike)
    dset.create_dataset("SNR", data=SNR_list)

    plt.figure()
    plt.hist(factor,bins=25)
    plt.xlabel('TD/FD speed up factor')
    plt.savefig(fp + 'hist_timing.png')

    mismatch = np.asarray(mismatch)
    plt.figure()
    plt.hist(np.log10(mismatch),bins=25)
    plt.xlabel('log10 Mismatch')
    plt.savefig(fp + 'mismatch.png')
    
    # np.save(fp +"failed_points.npy",np.asarray(failed_points))
    
    dset.close()
    return

if __name__ == "__main__":
    
    # set number of threads
    omp_set_num_threads(1)

    Tobs = args['Tobs'] # 1.05
    dt = args['dt'] #10.0
    eps = args['eps'] #1e-5

    waveform_kwargs = {
        "T": Tobs,
        "dt": dt,
        "eps": eps,
    }
    tot_numb = 50000
    fp = f"results/emri_T{Tobs}_seed{SEED}_dt{dt}_eps{eps}_fixedInsp{args['fixed_insp']}_tot_numb{tot_numb}_newprior"

    run_check(
        Tobs,
        dt,
        fp,
        emri_kwargs = waveform_kwargs,
        random_modes = False,
        get_fixed_inspiral = bool(args['fixed_insp']),
        tot_numb = tot_numb
    )
