# -*- coding: utf-8 -*-


import numpy as np


def groups_from_inds(inds):
    """Convert inds to group information

    Args:
        inds (dict): Keys are ``branch_names`` and values are inds
            np.ndarrays[ntemps, nwalkers, nleaves_max] that specify
            which leaves are used in this step.

    Returns:
        dict: Dictionary with group information.
            Keys are ``branch_names`` and values are
            np.ndarray[total number of used leaves]. The array is flat.

    """
    # prepare output
    groups = {}
    for name, inds_temp in inds.items():

        # shape information
        ntemps, nwalkers, nleaves_max = inds_temp.shape
        num_groups = ntemps * nwalkers

        # place which group each active leaf belongs to along flattened array
        group_id = np.repeat(
            np.arange(num_groups).reshape(ntemps, nwalkers)[:, :, None],
            nleaves_max,
            axis=-1,
        )

        # fill new information
        groups[name] = group_id[inds_temp]

    return groups


def get_acf(x, axis=0, fast=False):
    """
    Estimate the autocorrelation function of a time series using the FFT.
    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.
    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.
    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)
    """

    x = np.atleast_1d(x)
    m = [slice(None),] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2 ** np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x - np.mean(x, axis=axis), n=2 * n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[tuple(m)].real
    m[axis] = 0
    return acf / acf[tuple(m)]


def get_integrated_act(x, axis=0, window=50, fast=False, average=True):
    """
    Estimate the integrated autocorrelation time of a time series.
    See `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ on
    MCMC and sample estimators for autocorrelation times.
    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.
    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.
    :param window: (optional)
        The size of the window to use. (default: 50)
    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)
    """

    if axis != 0:
        # TODO: need to check this
        raise NotImplementedError

    # Compute the autocorrelation function.
    if isinstance(x, dict):
        is_dict = True
        ndim_total = 0
        values_out = []
        ind_breaks = []
        for name, values in x.items():
            nsteps, ntemps, nwalkers, nleaves_max, ndim = values.shape
            ndim_total += ndim
            ind_breaks.append(ndim_total)
            values_out.append(values.reshape(nsteps, ntemps, nwalkers, -1))

        x_in = np.concatenate(values_out, axis=-1)

    elif isinstance(x, np.ndarray):
        is_dict = False
        x_in = x
    else:
        raise ValueError("x must be dictionary of np.ndarrays or an np.ndarray.")

    f = get_acf(x_in, axis=axis, fast=fast)

    # Special case 1D for simplicity.
    if len(f.shape) == 1:
        return 1 + 2 * np.sum(f[1:window])

    # N-dimensional case.
    m = [slice(None),] * len(f.shape)
    m[axis] = slice(1, window)
    tau = 1 + 2 * np.sum(f[tuple(m)], axis=axis)

    if average:
        tau = np.average(tau, axis=1)

    if is_dict:
        splits = np.split(tau, ind_breaks, axis=-1)
        out = {name: split for name, split in zip(x.keys(), splits)}

    else:
        out = tau

    return out


def thermodynamic_integration_log_evidence(betas, logls):
    """
    Thermodynamic integration estimate of the evidence.

    This function origindated in ``ptemcee``.

    Args:
        betas (np.ndarray[ntemps]): The inverse temperatures to use for the quadrature.
        logls (np.ndarray[ntemps]): The mean log-Likelihoods corresponding to ``betas`` to use for
            computing the thermodynamic evidence.
    Returns:
        tuple:   ``(logZ, dlogZ)``: 
                Returns an estimate of the
                log-evidence and the error associated with the finite
                number of temperatures at which the posterior has been
                sampled.

    The evidence is the integral of the un-normalized posterior
    over all of parameter space:
    .. math::
        Z \\equiv \\int d\\theta \\, l(\\theta) p(\\theta)
    Thermodymanic integration is a technique for estimating the
    evidence integral using information from the chains at various
    temperatures.  Let
    .. math::
        Z(\\beta) = \\int d\\theta \\, l^\\beta(\\theta) p(\\theta)
    Then
    .. math::
        \\frac{d \\log Z}{d \\beta}
        = \\frac{1}{Z(\\beta)} \\int d\\theta l^\\beta p \\log l
        = \\left \\langle \\log l \\right \\rangle_\\beta
    so
    .. math::
        \\log Z(1) - \\log Z(0)
        = \\int_0^1 d\\beta \\left \\langle \\log l \\right\\rangle_\\beta
    By computing the average of the log-likelihood at the
    difference temperatures, the sampler can approximate the above
    integral.

    """

    # make sure they are the same length
    if len(betas) != len(logls):
        raise ValueError("Need the same number of log(L) values as temperatures.")

    # make sure they are in order
    order = np.argsort(betas)[::-1]
    betas = betas[order]
    logls = logls[order]

    betas0 = np.copy(betas)
    if betas[-1] != 0.0:
        betas = np.concatenate((betas0, [0.0]))
        betas2 = np.concatenate((betas0[::2], [0.0]))

        # Duplicate mean log-likelihood of hottest chain as a best guess for beta = 0.
        logls2 = np.concatenate((logls[::2], [logls[-1]]))
        logls = np.concatenate((logls, [logls[-1]]))
    else:
        betas2 = np.concatenate((betas0[:-1:2], [0.0]))
        logls2 = np.concatenate((logls[:-1:2], [logls[-1]]))

    # integrate by trapz
    logZ = -np.trapz(logls, betas)
    logZ2 = -np.trapz(logls2, betas2)
    return logZ, np.abs(logZ - logZ2)
