import argparse
# python check_mode_by_mode.py -Tobs 1.0 -dev 5 -eps 1e-3 -dt 10.0
parser = argparse.ArgumentParser(description='MCMC few')
parser.add_argument('-Tobs','--Tobs', help='Observation Time in years', required=True, type=float)
parser.add_argument('-dev','--dev', help='Cuda Device', required=True, type=int)
parser.add_argument('-eps','--eps', help='eps mode selection', required=True, type=float)
parser.add_argument('-dt','--dt', help='delta t', required=False, type=float)

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
from few.utils.utility import get_mu_at_t, get_p_at_t
from eryn.utils import TransformContainer
from few.utils.baseclasses import SchwarzschildEccentric
import time
from few.utils.utility import omp_set_num_threads
import h5py

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
    return_list=False,
    
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
    fixed_intrinsic=False,
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
                1: uniform_dist(np.log(1e-6), np.log(1e-4)),  # mass ratio
                2: uniform_dist(10.0, 15.0),  # p0
                3: uniform_dist(0.001, 0.7),  # e0
                4: uniform_dist(1.0, 1.000001),  # dist in Gpc
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

    from few.trajectory.inspiral import EMRIInspiral
    traj_module = EMRIInspiral(func="SchwarzEccFlux")

    dset = h5py.File(fp + '.h5',mode='w')
    factor = []
    mismatch = []
    failed_points = []
    list_injections = []
    timing_td = []
    timing_fd = []
    loglike = []
    
    tot_numb = 50000
    
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
        if fixed_intrinsic:
            injection_in[0] = 1e6
            injection_in[1] = 50
            injection_in[3] = 10.0
            injection_in[4] = 0.4
    
        M = injection_in[0]
        mu = injection_in[1]
        p0 = injection_in[3]
        e0 = injection_in[4]

        # get p in order to get an inspiral of 
        t_out = Tobs*1.001
        try:
            # run trajectory to get fixed inspiral
            if get_fixed_inspiral:
                p0 = get_p_at_t(traj_module,t_out,[M, mu, 0.0, e0, 1.0],index_of_p=3,index_of_a=2,index_of_e=4,index_of_x=5,traj_kwargs={},xtol=2e-6,rtol=8.881784197001252e-6,bounds=[6 + 2*e0+0.1, 40.0],)
                injection_in[3] = p0
            print('params ', M, mu, p0, e0)
            
            check = SchwarzschildEccentric()
            check.sanity_check_init(M,mu,p0,e0)

            #-------------------------
            tic = time.perf_counter()
            # generate FD waveforms
            data_channels_fd = few_gen(*injection_in, **emri_kwargs)
            # transform into hp and hc
            toc = time.perf_counter()
            fd_time = toc-tic
            #-------------------------
            # list version
            sig_fd = few_gen_list(*injection_in, **emri_kwargs)
            print("check 0 == ",xp.sum(sig_fd[0] - 1j *sig_fd[1] != data_channels_fd))

            # frequency goes from -1/dt/2 up to 1/dt/2
            frequency = few_gen.waveform_generator.create_waveform.frequency
            positive_frequency_mask = (frequency>=0.0)
            #-------------------------
            tic = time.perf_counter()
            # generate TD waveform, this will return a list with hp and hc
            data_channels_td = td_gen(*injection_in, **emri_kwargs)
            toc = time.perf_counter()
            td_time = toc-tic
            #-------------------------
            # list version
            sig_td = td_gen_list(*injection_in, **emri_kwargs)
            fft_td_wave_p = xp.fft.fftshift(xp.fft.fft(sig_td[0])) * dt
            fft_td_wave_c = xp.fft.fftshift(xp.fft.fft(sig_td[1])) * dt
            sig_td = [fft_td_wave_p,fft_td_wave_c]
            
            # store timing
            timing_td.append(td_time)
            timing_fd.append(fd_time)
            list_injections.append(injection_in)
            print("TD/FD time",td_time/fd_time, "TD", td_time, "FD", fd_time )
            factor.append(td_time/fd_time)

            # kwargs for computing inner products
            fd_inner_product_kwargs = dict( PSD="cornish_lisa_psd", use_gpu=use_gpu, f_arr=frequency[positive_frequency_mask])
            # check no list
            nolist_check = inner_product(
                data_channels_fd[positive_frequency_mask],
                xp.fft.fftshift(xp.fft.fft(data_channels_td))[positive_frequency_mask] * dt, 
                normalize=True, **fd_inner_product_kwargs)
            print("must be approximately 1~",nolist_check)

            sig_fd = [sig_fd[0][positive_frequency_mask],sig_fd[1][positive_frequency_mask]]
            sig_td = [sig_td[0][positive_frequency_mask],sig_td[1][positive_frequency_mask]]
            
            # mismatch
            Mism = np.abs(1-inner_product(sig_fd, sig_td, normalize=True, **fd_inner_product_kwargs))
            print("mismatch total and partial ", Mism)
            if Mism>0.01:
                mask_non_zero = (sig_fd[0]!=complex(0.0))
                plt.figure(); plt.loglog( xp.abs(sig_fd[0][mask_non_zero]).get()**2 ); plt.loglog( xp.abs(sig_td[0][mask_non_zero]).get()**2 ); plt.savefig(f'high_mism/mism{Mism}.png')

            if use_gpu:
                mismatch.append(Mism.get())
            else:
                mismatch.append(Mism)
            
            # loglike
            sig_inner = [sig_fd[0]-sig_td[0],sig_fd[1]-sig_td[1]]
            if use_gpu:
                logl = -0.5 * inner_product(sig_inner, sig_inner, normalize=False, **fd_inner_product_kwargs).get()
            else:
                logl = -0.5 * inner_product(sig_inner, sig_inner, normalize=False, **fd_inner_product_kwargs)
            print("logl ", logl)
            loglike.append(logl)

        except:
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
    loglike
    ]
    for el in to_store:
        el = np.asarray(el)
    dset.create_dataset("mismatch", data=mismatch)
    dset.create_dataset("failed_points", data=failed_points)
    dset.create_dataset("list_injections", data=list_injections)
    dset.create_dataset("timing_td", data=timing_td)
    dset.create_dataset("timing_fd", data=timing_fd)
    dset.create_dataset("loglike", data=loglike)

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
    omp_set_num_threads(4)
    Tobs = args['Tobs'] # 1.05
    dt = args['dt'] #10.0
    eps = args['eps'] #1e-5

    waveform_kwargs = {
        "T": Tobs,
        "dt": dt,
        "eps": eps,
    }

    fp = f"emri_T{Tobs}_seed{SEED}_dt{dt}_eps{eps}_"

    run_check(
        Tobs,
        dt,
        fp,
        emri_kwargs = waveform_kwargs,
        random_modes = True,
        get_fixed_inspiral = False,
    )
