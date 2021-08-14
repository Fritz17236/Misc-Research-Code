import os

import numpy as np
import math


from constants import TIME_END_DEFAULT, TIME_BEGIN_DEFAULT, DT_DEFAULT, BIN_WIDTH_DEFAULT
import scipy.io as spio
import tqdm
from scipy.signal import welch
import matplotlib.pyplot as plt


def get_bin_edges(t_start, t_end, width):
    """
    Get bin edges of a given width an stride between start and end times.

    :param t_start:
    :param t_end:
    :param width:
    :return:
    """
    return np.arange(start=t_start, stop=t_end, step=width)


def load_mat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check ks to cure all entries
    which are still mat-objects
    '''

    def _check_ks(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for k in d:
            if isinstance(d[k], spio.matlab.mio5_params.mat_struct):
                d[k] = _todict(d[k])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_ks(data)


def bin_spikes(spike_train, bin_edges):
    """
    Count the spikes that occur within bins

    :param spike_train: array of spike times to bin
    :param bin_edges: edges of bins.
    :return: counts array such that counts[0] = # spikes between bin_edges[0], bin_edges[1].
    """
    counts = np.zeros(bin_edges.shape)

    try:
        num_spikes = len(spike_train)
    except TypeError as te:
        return counts

    return np.histogram(spike_train, bins=bin_edges)[0]


def get_session_firing_rates(session, t_start=TIME_BEGIN_DEFAULT, t_end=TIME_END_DEFAULT, bin_width=BIN_WIDTH_DEFAULT):
    """
    Compute the Session Firing Rate.

    Return
    ts, (num_ts, num_trials, num_neurons) data
    """
    print("Computing Firing Rates...")
    spike_times = session['neuron_single_units']
    num_trials = len(spike_times[0])
    num_neurons = len(spike_times)

    bin_edges = get_bin_edges(t_start=t_start, t_end=t_end, width=bin_width)
    ts = bin_edges[1:] - bin_width # causal estimation (rate computed looking back)
    frs = np.zeros((len(ts), num_trials, num_neurons))

    for i in tqdm.tqdm(range(num_trials)):
        for j in range(num_neurons):
            try:
                sts = spike_times[j][i]
                frs[:, i, j] = bin_spikes(sts, bin_edges)[:len(ts)] / bin_width
            except IndexError:
                pass # Pass for now to leave as zeros; filtering happens at later stage

    print("done.")
    return ts, frs
# region old code


def kernel_gaussian(t, sigma=.05):
    '''
    A gaussian window function
    t: (numpy 1d array) time to evaluate funciton
    sigma: (float) width of Gaussian curve
    return: (numpy 1d array) function evaluated at each point in t
    '''
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(
        -np.square(t / (np.sqrt(2) * sigma))
    )


def kernel_square(t, width=.05):
    '''
    A square window function
    t: (numpy 1d array) time to evaluate funciton
    width: (float) width of the window (end to end)
    return: (numpy 1d array) function evaluated at each point in t
    '''
    output = np.ones(t.shape)
    output[t > width / 2] = 0
    output[t < - width / 2] = 0
    return output / width


def kernel_causal_alpha(t, width=.05):
    '''
    A square window function
    t: (numpy 1d array) time to evaluate funciton
    width: (float) temporal resolution = 1 / alpha
    return: (numpy 1d array) function evaluated at each point in t
    '''
    alpha = 1 / width
    output = alpha ** 2 * t * np.exp(- alpha * t)
    output[output < 0] = 0
    return output


def estimate_spike_rate(spike_times, kernel, dt=.001, t_start=TIME_BEGIN_DEFAULT, t_end=TIME_END_DEFAULT):
    '''
    Estimate the spike rate from a sparse array of spike times

    Args:
    spike_times: (numpy 1d array) sparse vector of neuron spike times
    kernel: func(float--> float) function applied to spike train to generate smooth time series (see Dayan & Abbot)
    dt: (float) time step between data points (inverse of sample rate)
    return: (numpy 1d array) dense time series of estimated spike rate
    '''
    ts, rho = densify_spike_times(spike_times, sample_rate=dt, t_start=t_start, t_end=t_end)
    conv = np.convolve(kernel(ts), rho, mode='same')

    return ts, conv


def densify_spike_times(spike_times, sample_rate=.001, t_start=TIME_BEGIN_DEFAULT, t_end=TIME_END_DEFAULT,
                        verbose=False):
    '''
    Given a sparse array of spike arrival times, densify them to a continuous time series sampled at dt with zero padding

    Args:
    spike_times: (numpy 1d array) sparse vector of neuron spike times
    dt: (float) time step between data points (inverse of sample rate)
    return: (numpy 1d array) dense time series with 1 where spikes occur and zero everywhere else
    '''
    ts = np.arange(start=t_start, stop=t_end, step=sample_rate)
    rho = np.zeros(ts.shape)

    try:
        num_spikes = len(spike_times)
    except TypeError as te:
        if verbose:
            print("Trial has 1 spike or less, skipping...")
        return ts, rho

    idxs = np.asarray(np.digitize(spike_times, ts))
    idxs[idxs >= len(rho)] = 0

    idxs = np.trim_zeros(idxs)

    try:
        assert (np.all(np.diff(idxs) != 0)), "Two spikes occurred within dt = {0} seconds of each other".format(sample_rate)
    except AssertionError:
        if verbose:
            print("Warning: Unrealistic inter-spike interval detected. Returning no-spike time series")
        return ts, rho

    rho[idxs] += 1
    return ts, rho


def cross_correlation(x, y):
    '''
    Compute the autocorrelation of a two functions with time lag

    Args:
    x: (numpy 1d array) function to compute cross-correlation of
    y: (numpy 1d array) function to compute cross-correlation of
    '''
    x_demeaned = x - np.mean(x)
    x_norm_squared = x_demeaned.T @ x_demeaned

    y_demeaned = y - np.mean(y)
    y_norm_squared = y_demeaned.T @ y_demeaned
    try:
        corr = np.correlate(x_demeaned, y_demeaned, mode='same') / np.sqrt(x_norm_squared * y_norm_squared)
        assert (len(corr) == len(x) == len(
            y)), "Correlated Output length (length = {0}) should match input vector lengths of " \
                 "x (length = {1}) and y (length = {2}).".format(len(corr), len(x), len(y))

        return corr

    except FloatingPointError:
        raise


def autocorrelation(x):
    '''
    Compute the autocorrelation of a function with time lag

    Args:
    x: (numpy 1d array) function to compute autocorrelation of
    '''
    #     x_demeaned = x - np.mean(x)
    #     x_norm_squared = x_demeaned.T @ x_demeaned
    #     return np.correlate(x_demeaned,x_demeaned,mode='same') / x_norm_squared
    return cross_correlation(x, x)


def ts_to_acorr_lags(ts):
    '''
    Given a set of times for some vector, return a vector of autocorrelation lags to use for plotting
    the autocorrelation function of that vector

    Args:
    ts:(numpy 1d array) vector of times of each point in some vector x, monotonic and evenly spaced
    '''
    t_start = ts[0]
    t_end = ts[-1]
    N = len(ts)
    dt = ts[1] - ts[0]
    assert np.all(np.isclose(np.diff(ts), dt)), "Times ts are not evenly spaced and/or monotonic"

    lags = np.zeros((N,))

    if N % 2 == 0:
        # if x even, lags = [-dt * N//2, 0] and [0, dt * (N//2 -1)]
        lags = np.linspace(start=-dt * (N // 2), stop=dt * (N // 2 - 1), num=N)
    else:
        # if x odd, go [-dt * N//2, 0 , dt * N//2]
        lags = np.linspace(start=-dt * (N // 2), stop=dt * (N // 2), num=N)

    return lags


def estimate_spike_rate_classic(spike_times, bin_width=.05, stride=.01):
    '''
    Use non-overlapping bins to estimate spike rate.
    '''
    bin_centers, fr = sliding_histogram(
        spike_times,
        TIME_BEGIN_DEFAULT,
        TIME_END_DEFAULT,
        bin_width=bin_width,
        stride=stride,
        rate=True
    )  # fr: (n_bins, n_trials, n_neurons), bin_centers = n_bins


def get_trial_neuron_firing_rates(arr_spike_rates):
    num_neurons = len(arr_spike_rates)
    num_trials = len(arr_spike_rates[0])
    firing_rates = np.zeros((num_neurons, num_trials))

    for i in range(num_neurons):
        for j in range(num_trials):
            firing_rates[i][j] = np.sum(arr_spike_rates[i][j] != 0)
    return firing_rates


def get_neuron_idx_max_firing_rate(arr_spike_rates):
    firing_rates = get_trial_neuron_firing_rates(arr_spike_rates)

    avg_firing_rates = np.mean(firing_rates, axis=1)
    max_firing_rate_idx = np.argsort(avg_firing_rates)[-1]
    return max_firing_rate_idx


def sliding_histogram(spikeTimes, begin_time, end_time, bin_width, stride, rate=True):
    '''
    Calculates the number of spikes for each unit in each sliding bin of width bin_width, strided by stride.
    begin_time and end_time are treated as the lower- and upper- bounds for bin centers.
    if rate=True, calculate firing rates instead of bin spikes.
    '''
    bin_begin_time = begin_time
    bin_end_time = end_time
    # This is to deal with cases where for e.g. (bin_end_time-bin_begin_time)/stride is actually 43 but evaluates to 42.999999... due to numerical errors in floating-point arithmetic.
    if np.allclose((bin_end_time - bin_begin_time) / stride, math.floor((bin_end_time - bin_begin_time) / stride) + 1):
        n_bins = math.floor((bin_end_time - bin_begin_time) / stride) + 2
    else:
        n_bins = math.floor((bin_end_time - bin_begin_time) / stride) + 1
    binCenters = bin_begin_time + np.arange(n_bins) * stride

    binIntervals = np.vstack((binCenters - bin_width / 2, binCenters + bin_width / 2)).T

    # Count number of spikes for each sliding bin
    binSpikes = np.asarray([[[np.sum(np.all([trial >= binInt[0], trial < binInt[1]], axis=0)) for binInt in
                              binIntervals] for trial in unit]
                            for unit in spikeTimes]).swapaxes(0, -1)

    if rate:
        return binCenters, binSpikes / float(bin_width)  # get firing rates
    return binCenters, binSpikes


def get_psd(x, fs):
    """
    Compute Power Spectral Density of x,
    returns |X|^2 same legnth of x
    """
    # X = np.fft.rfft(x, n=len(x))
    # Pxx = X * np.conj(X)
    # freqs = np.fft.fftfreq(n=len(x), d=1 / fs)
    assert(x.ndim==1), "Given vector should only 1 dimension but had shape {0}".format(x.shape)
    freqs, Pxx = welch(x, fs=fs, nfft=len(x), detrend=None, scaling='spectrum', return_onesided=True)

    return freqs, Pxx


def get_whitening_filter(x, fs, b=np.inf, mode='highpass'):
    """
    Get a frequency-domain filter that whitens the power spectrum of x,
    i.e. the multiplicative inverse of the power spectral density of x

    Returns freqs, filt
    """
    from scipy.signal import welch

    #     freqs, Pxx = scipy.signal.welch(x, fs=fs, nfft=len(x), detrend=None, scaling='spectrum')
    #     filt = 1 / np.sqrt(Pxx)
    #     filt[freqs > b] = 0
    #     return freqs, filt
    assert (x.ndim == 1), "Given vector should only 1 dimension but had shape {0}".format(x.shape)
    freqs, psd = get_psd(x, fs)
    psd[np.isclose(psd, 0)] = 1
    filt = 1 / np.sqrt(psd)

    # if mode == 'highpass':
    #     filt[freqs > b] = 1
    # elif mode == 'lowpass':
    #     filt[freqs <= b] = 1
    # else:
    #     print("Whitening Mode option \'{0}\' not valid, reverting to highpass default".format(mode))
    return freqs, filt


def apply_filter(x, Filter):
    """
    Apply frequency domain Filter to real-valued time domain signal x.
    Assumes Filter has same number of points as x.
    Returns filtered x series
    """

    X = np.fft.rfft(x.copy())
    assert (len(Filter) == len(
        X)), "Filter ({0} points) should have same number of points as  rfft of time series ({1} points).".format(
        len(Filter), len(X)
    )
    X *= Filter

    out_filtered = np.fft.irfft(X, n=len(x))

    assert (len(out_filtered) == len(
        x)), "Filtered Output (length = {0}) should have same length as input (length = {1}).".format(
        len(out_filtered), len(x)
    )
    return out_filtered


def get_whitened_cross_correlation(x, y, fs, bandwidth=np.inf, mode='highpass'):
    """
    Whiten the spectrum of x, apply same filter to y and compute cross-correlation.
    Returns the filtered cross correlation.
    return: lags, cross_corr
    """

    freqs, Filt = get_whitening_filter(x, fs=fs, b=bandwidth, mode=mode)
    x_filtered = apply_filter(x.copy(), Filt)
    y_filtered = apply_filter(y.copy(), Filt)

    cross_corr = cross_correlation(x_filtered, y_filtered)
    assert (len(cross_corr) == len(
        x)), "Cross-Correlated Output (length {0}) should have same length as input vectors (length {1}).".format(
        len(cross_corr), len(x))
    return cross_corr


def get_trials_by_lick_direction(session, ldir):
    """
    Get the trial indices of those trials where mouse was tasked to lick a given direction either left 'l' or right 'r'.
    """
    assert (
            ldir == 'l' or ldir == 'r'), "Lick Direction ldir = \'{0}\' must be wither left (\'l\') or right (\'r\').".format(
        ldir)
    return np.squeeze(np.argwhere(session['task_trial_type'] == ldir))

#
# def _get_session_firing_rates(session, t_start=TIME_BEGIN_DEFAULT, t_end=TIME_END_DEFAULT, bin_width=BIN_WIDTH_DEFAULT):
#     """
#     Compute the Session Firing Rate.
#
#     Return
#     ts, (num_ts, num_trials, num_neurons) data
#     """
#     print("Computing Firing Rates...")
#     spike_times = session['neuron_single_units']
#     num_trials = len(spike_times[0])
#     num_neurons = len(spike_times)
#
#     ts = get_bin_edges(t_start=t_start, t_end=, width=) - bin_width
#     frs = np.zeros((len(ts), num_trials, num_neurons))
#
#     for i in tqdm.tqdm(range(num_trials)):
#         for j in range(num_neurons):
#             _, frs[:, i, j] = estimate_spike_rate(spike_times[j][i], kernel, dt=bin_width, t_start=TIME_BEGIN_DEFAULT,
#                                                   t_end=TIME_END_DEFAULT)
#     print("done.")
#     return ts, frs




def filter_by_firing_rate(firing_rates, threshold=1):
    """
    Filter the firing rates in the session data based on their trial-averaged firing rates.

    :param threshold: Firing rate below which firing rates are discarded
    :param firing_rates: firing rates of neurons to filter
    """
    firing_rates_mean_trial_bin = np.mean(firing_rates, axis=(0,1))
    keep_neurons = firing_rates_mean_trial_bin > threshold
    firing_rates_filtered = firing_rates[:, :, keep_neurons != 0]
    assert(firing_rates_filtered.shape[2] == sum(keep_neurons != 0))
    return firing_rates_filtered, keep_neurons

#endregion

def spike_train_cch(trig, ref):
    """
    Compute the spike train cross-correlation histogram

    :param trig: trigger spike train array
    :param ref:  reference spke train array
    :return: (lags, counts)
    """



    #densify to get time series of each spike train
        # values are 0 or 1 at given spike times

    # dilute spike trains

    # get times

    #split time into +/- Bins of width b


    # for each bin m look for spikes between m*b - b/2 : m*b + b/2
        #cch is number of spikes from trigger spike followed by reference spike
        #e.g. B = .001,  m = 0, then cch is number of spike pairs where trigger is follow by reference within .5 ms

    # for m in range(-M, +M) (inclusive)
        # for each trigger_spike in trigger_spike_train
            # for each reference_spike in reference_spike_train
                # if m*B - B/2 < time(reference_spike) < m*b + B/2:
                    # cch(m) += 1
