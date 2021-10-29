"""
utils.py Various utility functions, not all currently used.

"""
from time import time

import numpy
import numpy as np
import math

import scipy.signal
from sklearn.decomposition import PCA

import constants
from constants import TIME_END_DEFAULT, TIME_BEGIN_DEFAULT, BIN_WIDTH_DEFAULT
import matplotlib.pyplot as plt
import scipy.io as spio
from tqdm import tqdm
from scipy.signal import welch
import decimal
from scipy.signal import convolve
from scipy.stats import poisson


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
    ts = bin_edges[1:] - bin_width  # causal estimation (rate computed looking back)
    frs = np.zeros((len(ts), num_trials, num_neurons))

    for i in tqdm(range(num_trials)):
        for j in range(num_neurons):
            try:
                sts = spike_times[j][i]
                frs[:, i, j] = bin_spikes(sts, bin_edges)[:len(ts)] / bin_width
            except IndexError:
                pass  # Pass for now to leave as zeros; filtering happens at later stage

    print("done.")
    return ts, frs


def plot_data(x, y, **kwargs):
    """

    :param x: 1d numpy array specifying the independent variable (x-axis) to plo9t
    :param y: either 1d array specifying dependent variable (y-axis) or if two_dim=True, is a two-dimensional
        numpy array, and second axis (with length y.shape[1]) is averaged.
    :param title:
    :param kwargs:
        two_dim: true if plotting a 2d numpy array with averaging
        fill_error: if two_dim is true, then the standard deviation and +/- the Standard error of the mean is plotted
            around the mean
        color: specify the color of a curve (string)
        fig_name: name of the figure generated, also can be used to plot multiple figures at once (string)
        xlabel: set the x-axis label (string)
        ylabel: set the y-axis label (string)
        xlim: set x limits (2-tuple or 2-list)
        ylim: set y limits(2-tuple or 2-list)
        legend: add a legend (bool)
        save_path: save the figure at a given path (string)
        title: set the figure title (string)
        show: show the figure (bool)
        label: label the plotted time series
        line_at_zero: insert a vertical line at x=0
        figsize: specify the size of the figure (units?)
        vline: draw a vertical line at a given x value
        hlin: draw a horizontal line at a given y value

    :return: figure : handle to the generated figure
    """
    valid_args = ["two_dim", "fill_error", "color",
                  "fig_name", "xlabel", "ylabel",
                  "xlim", "ylim", "legend",
                  "save_path", "title", "show", "label", "vline", "hline", "figsize"]

    for val in kwargs.keys():
        assert (val in valid_args), "Keyword Argument \'{0}\' not a valid argument".format(val)

    assert (x.shape[0] == y.shape[0]), "x and y to have compatible shapes, but first dimensions" \
                                       " are {0} and {1}".format(x.shape[0], y.shape[0])

    if "title" in kwargs:
        fig = plt.figure(kwargs["title"])
        plt.title(kwargs["title"])

    else:
        fig = plt.figure()

    if "figsize" in kwargs:
        fig.set_size_inches(kwargs["figsize"])

    if "two_dim" in kwargs:
        assert (x.shape[0] == y.shape[0]), "x and y expected to have compatible shapes, but first dimensions" \
                                           " are {0} and {1}".format(x.shape[0], y.shape[0])
        lines = plt.plot(x, np.mean(y, axis=1))

    else:
        assert (x.shape == y.shape), "x and y expected to have same shapes but shape(x)={0} and shape(y)={1}" \
                                     "".format(x.shape, y.shape)
        lines = plt.plot(x, y)

    if "fill_error" in kwargs:
        if "two_dim" in kwargs:
            error = np.std(y, axis=1) / np.sqrt(y.shape[1])
            y = np.mean(y, axis=1)
            lines.append(plt.fill_between(x, y - error, y + error, alpha=.3))
        else:
            lines.append(plt.fill_between(x, y - kwargs["fill_error"], y + kwargs["fill_error"], alpha=.3))

    if "vline" in kwargs:
        plt.axvline(x=kwargs["vline"], c='k')

    if "hline" in kwargs:
        plt.axhline(y=kwargs["hline"], c='k')

    if "color" in kwargs:
        for line in lines:
            line.set_color(kwargs["color"])

    if "label" in kwargs:
        lines[0].set_label(kwargs["label"])

    if "xlabel" in kwargs:
        plt.xlabel(kwargs["xlabel"])

    if "ylabel" in kwargs:
        plt.ylabel(kwargs["ylabel"])

    if "xlim" in kwargs:
        plt.xlim(kwargs["xlim"])

    if "ylim" in kwargs:
        plt.ylim(kwargs["ylim"])

    if "legend" in kwargs:
        plt.legend()

    if "save_path" in kwargs:
        plt.savefig(kwargs["save_path"], dpi=128, bbox_inches='tight')

    if "show" in kwargs:
        plt.show()


def pad_spike_times(spike_times, max_num_spikes=None):
    """
    Pad a ragged spike times array to a 3d numpy array of fixed size

    :param spike_times: nested ragged array of shape [num_neurons][num_trials][variable num spikes]
    :param max_num_spikes: specify a pad width along the number of spikes dimension (3). If none, it's computed as the
    maximum number of spikes within spike times along all trials and neurons.
    :return: spike_times_padded: 3d array of shape [num_neurons][num_trials][max_num_spikes]
    """
    num_neurons = spike_times.shape[0]
    num_trials = spike_times.shape[1]

    # def boolean_indexing(v, fillval=np.nan):
    #     out = np.full(mask.shape, fillval)
    #     out[mask] = np.concatenate(v)
    #     return out
    # return boolean_indexing(spike_times, fillval=0)

    if not max_num_spikes:
        max_num_spikes = max([len(spike_times[i][j]) for i in range(num_neurons) for j in range(num_trials)])

    import time
    start = time.time()

    spike_times_padded = np.zeros((num_neurons, num_trials, max_num_spikes))
    mask = spike_times.nonzero()
    print(mask.shape)
    print(spike_times.shape)
    print(spike_times_padded.shape)

    spike_times_padded[mask] = spike_times

    end = time.time()
    print("Time to pad = {0}".format(end - start))
    exit(0)

    return spike_times_padded
    for idx_trial in range(num_trials):
        for idx_neuron in range(num_neurons):
            arr_len = len(spike_times[idx_neuron][idx_trial])
            spike_times_padded[idx_neuron, idx_trial, :arr_len] = spike_times[idx_neuron][idx_trial]

    return spike_times_padded


def dilute_spike_train(train, bound=constants.SPIKE_TRAIN_DILUTION_BOUND):
    """
    Dilute a spike train such that the interspike intervals are bounded from below

    :param train: (numpy array) sparse numpy array of spike times to dilute
    :param bound: (float) lower bound of interspike intervals (distance between subsequent spikes
    :return: diluted spike train
    """
    try:
        l = len(train)
        if len(train) < 2:
            return np.asarray([0])
    except(AttributeError):
        return np.asarray([0])

    diluted = [train[0]]
    for idx_spike in range(1, len(train)):
        if train[idx_spike] - diluted[-1] < bound:
            continue
        else:
            diluted.append(train[idx_spike])
    if len(diluted) < 2:
        return np.asarray([0])
    assert(np.min(np.diff(diluted) >= bound))

    return np.asarray(diluted)


def spike_train_cch_raw(trig, ref, rng, bin_width):
    """
    Compute the spike train cross-correlation histogram

    :param trig: trigger spike train array
    :param ref:  reference spke train array
    :param bin_width: (s) the width of the histogram bins
    :param rng: (int) number of bins to compute cch over, cch lag times are in the interval +/- bin_width * range
    :return: (lags, counts), length M+1 where M is rng
    """

    # compute raw cch
    cch1 = np.zeros((2 * rng + 1,), dtype=np.int)
    cch2 = np.zeros((2 * rng + 1,), dtype=np.int)

    start = time()
    trig = dilute_spike_train(trig)
    ref = dilute_spike_train(ref)



    T = np.max([trig[-1], ref[-1]]) - np.min([trig[0], ref[0]])
    assert(np.all(T >= trig - trig[0]) and np.all(T >= ref - ref[0]))

    sample_debias_count = int(T//bin_width - rng)


    for spike_1 in trig[:sample_debias_count]:
        bin_edges = spike_1 + bin_width * (np.arange(-rng - 1, rng + 1) + 1)
        cch1 += np.histogram(a=ref, bins=bin_edges, range=(np.min(bin_edges), np.max(bin_edges)))[0]
    for spike_2 in ref[:sample_debias_count]:
        bin_edges = spike_2 + bin_width * (np.arange(-rng - 1, rng + 1) + 1)
        cch2 += np.histogram(a=trig, bins=bin_edges, range=(np.min(bin_edges), np.max(bin_edges)))[0]

    left_half = np.flip(cch2[1 + len(cch2)//2:])
    right_half = cch1[len(cch1)//2:]
    assert(len(left_half) == rng), "Left half should have {0} points but had {1}".format(rng, len(left_half))
    assert(len(right_half) == rng + 1), "Right half should have {0} points but had {1}".format(rng+1, len(right_half))
    cch = np.hstack((left_half, right_half))
    assert(len(cch) == 2 * rng + 1), "output cch should have len {0} but had len {1}".format(2*rng+1, len(cch))
    # cch = np.zeros((2 * rng + 1,))
    # for spike in trig:
    #     bin_edges = spike + bin_width * (np.arange(-rng - 1, rng + 1) + 1)
    #     cch += np.histogram(a=ref, bins=bin_edges, range=(np.min(bin_edges), np.max(bin_edges)))[0]

    return cch


def get_cch_predictor(trigs, refs, rng, window_width, bin_width, trial_len):
    """
    Compute the predictor function for a given cch
    :param bin_width:
    :param trigs: [num_trials][var_num_spikes] trigger spikes trians over [num_trials]
    :param refs:  [num_trials][var_num_spikes] reference spike trians over [num_trials]
    :param rng: (int) number of bins extending in positive direction (giving  2 * rng + 1 bins total)
    :param window_width: (even int) width of convolution window used to compute p1 and p5 values, sets temporal precision
    of estimate to 2 * window_width. window width is given as an integer multiple of bin_width (2 * bin_width, etc. )
    should be even since window_width / 2 bins are padded on cch edges to address boundary effects
    :return: cch_pred (1d numpy array) a predictor for determining probability values
    """
    assert (window_width % 2 == 0), "Window width_must be even but as {0}".format(window_width)
    # get total cch
    cch = np.zeros((2*rng+1,))
    num_trials = trigs.shape[0]
    for j in range(num_trials):
        cch += spike_train_cch_raw(trigs[j], refs[j], rng, bin_width)


    # convolve to get predictor
    window = np.ones((window_width,)) / window_width  # divide so window has unit area
    prepend = np.flip(cch[1: 1 + window_width // 2].copy())
    append = np.flip(cch[-window_width // 2:].copy())
    assert (len(prepend) == len(append) == window_width // 2), "len prepend {0} should equal len append {1}" \
                                                               " should equal window_length / 2 {2}".format(
        len(prepend), len(append), window_width // 2)

    cch_padded = np.hstack((prepend, cch, append))
    # scipy.signal.windows.gaussian()
    for j in range(window_width // 2):
        assert (prepend[j] == cch[window_width // 2 - j])
        assert (append[j] == cch[-j - 1])

    cch_predictor = scipy.signal.convolve(cch_padded, window, mode='full')
    center = rng + window_width
    assert (center == int(
        len(cch_predictor) // 2)), "cch predictor should have center idx of {0}, but has length {1}".format(
        center, len(cch_predictor) // 2)
    cch_predictor = cch_predictor[center - rng: center + rng + 1]

    assert (len(cch_predictor) == len(cch)), "len predictor {0} and len actual cch {1} should be the same ".format(
        len(cch_predictor), len(cch)
    )

    return cch_predictor / (num_trials * trial_len)
    # densify to get time series of each spike train
    # values are 0 or 1 at given spike times

    # dilute spike trains

    # get times

    # split time into +/- Bins of width b

    # for each bin m look for spikes between m*b - b/2 : m*b + b/2
    # cch is number of spikes from trigger spike followed by reference spike
    # e.g. B = .001,  m = 0, then cch is number of spike pairs where trigger is follow by reference within .5 ms

    # for m in range(-M, +M) (inclusive)
    # for each trigger_spike in trigger_spike_train
    # for each reference_spike in reference_spike_train
    # if m*B - B/2 < time(reference_spike) < m*b + B/2:
    # cch(m) += 1


def get_p_vals(cch, cch_predictor):
    """
    Compute the probability of observed cch given cch predictor

    :param cch: (1d numpy array) cross-correlation historgram, output from spike_train_cch_raw
    :param cch_predictor: (1d numpy arrray) cch predctor, output from get_cch_predictor
    :return:
    """

    ps = np.zeros((len(cch_predictor),))

    def poisson(lam, cnt):
        curr_sum = 0
        for idx_count in np.arange(cnt):
            top = np.exp(-lam) * lam ** idx_count
            log_bot = idx_count * np.log(idx_count) - idx_count  # stirling's approximation
            curr_sum += top / np.exp(log_bot)
        return curr_sum
        # return 1 - float(np.sum([
        #     decimal.Decimal(np.exp(-lam) * lam ** idx_count) / decimal.Decimal(np.math.factorial(idx_count))
        #     for idx_count in np.arange(cnt)
        # ]))

    ps = 1 - np.asarray([scipy.stats.poisson.cdf(k=cch[idx_bin]-1, mu=cch_predictor[idx_bin]) for idx_bin in range(len(cch))])
    # ps[ps < 0] = 0
    # ps[np.isnan(ps)] = 0
    assert (
        ((np.min(ps) >= 0) or np.isclose(np.min(ps), 0))
        and
        np.all(ps <= 1)
    ), "Probabilities in invalid range: min={0}, max={1}".format(ps.min(), ps.max())
    return ps


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
        assert (np.all(np.diff(idxs) != 0)), "Two spikes occurred within dt = {0} seconds of each other".format(
            sample_rate)
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
    x_copy = x.copy()
    y_copy = y.copy()

    x_demeaned = x_copy - np.mean(x_copy)
    x_norm_squared = np.square(x_demeaned).sum()
    y_demeaned = y_copy - np.mean(y_copy)
    y_norm_squared = np.square(y_demeaned).sum()

    try:
        corr = np.correlate(x_demeaned, y_demeaned, mode='same') / np.sqrt(x_norm_squared * y_norm_squared)
        assert (len(corr) == len(x) == len(
            y)), "Correlated Output length (length = {0}) should match input vector lengths of " \
                 "x (length = {1}) and y (length = {2}).".format(len(corr), len(x), len(y))
        corr /= (len(x) - np.abs(np.arange(-len(x) // 2, len(x) // 2)))
        corr *= len(x)

        return corr

    except FloatingPointError:
        raise

    except ValueError:
        raise


# test cross correlation
# first check for two single gaussians,

# then check by computing over several gaussians  and vaeraging

# then average several gaussians & compute


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
    assert (x.ndim == 1), "Given vector should only 1 dimension but had shape {0}".format(x.shape)
    freqs, Pxx = welch(x, fs=fs, nfft=len(x), nperseg=len(x) // 4, detrend=None, scaling='spectrum',
                       return_onesided=True)

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


def filter_by_region(arr, filter_axis, regions, target_region):
    """
    Get a boolean mask which is true for neurons in region, false otherise
    :param neuron_regions:
    :param region:
    :return:
    """
    try:
        len(regions)
    except TypeError:
        print("regions is must have length, but was a scalar or none")

    assert filter_axis < arr.ndim
    assert len(regions) == arr.shape[filter_axis], "{0} != {1}".format(len(regions), len(arr.shape[filter_axis]))
    mask = regions == target_region
    return arr[:, :, mask]
    # return np.take(arr, mask, axis=filter_axis)


def compute_trial_isis(session, ts, region='all'):
    """
    Compute the interspike interval vs time sample at points dt

    :param session: Session instance
    :param ts: time steps to sample isi data
    :param region: restrict to a specific region
    :return: isis (numpy 3d array of shape [num_ts][num_trials][num_neurons] the sampled interspike intervals
    """
    num_trials = session.get_num_trials()
    num_neurons = session.get_num_neurons()
    spike_times = session.get_spike_times()

    print("\Computing & Sampling ISI's...")
    isis = np.zeros((len(ts), num_trials, num_neurons))
    for idx_neuron in tqdm(np.arange(num_neurons)):
        for idx_trial in np.arange(num_trials):

            train = spike_times[idx_neuron][idx_trial]
            train = train[(train < ts[-1]) & (train > ts[0])]

            try:
                train[0]
            except IndexError:
                continue

            prev_spike = train[0]
            for spike_time in train[1:]:
                idx_spike_ts = np.nonzero(ts >= spike_time)[0]
                try:
                    idx_spike_ts = idx_spike_ts[0]
                except IndexError:
                    continue
                isis[idx_spike_ts:, idx_trial, idx_neuron] = spike_time - prev_spike
                prev_spike = spike_time
    isis = filter_by_region(arr=isis, filter_axis=2, regions=session.get_neuron_brain_regions(), target_region=region)
    return isis


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
    firing_rates_mean_trial_bin = np.mean(firing_rates, axis=(0, 1))
    keep_neurons = firing_rates_mean_trial_bin > threshold
    firing_rates_filtered = firing_rates[:, :, keep_neurons != 0]
    assert (firing_rates_filtered.shape[2] == sum(keep_neurons != 0))
    return firing_rates_filtered, keep_neurons

# endregion
def filter_firing_rates_by_time(frs,ts, t_starts, t_ends):
    """
    truncate firing rates to fall between two sets of times.
    :param frs: (num_bins, num_trials, num_neurons) firing rate data
    :param t_starts: (num_trials, 1) start times
    :param t_ends:  (num_trials, 1) end times
    :return: filtered_frs (num_neurons, num_bins_stacked) concatenated filtered firing rates
    """

    for idx_trial, t_start in enumerate(t_starts):
        t_end = t_ends[idx_trial]
        mask = np.argwhere( (ts > t_start) & (ts < t_end))
        slice = frs[mask, idx_trial, :].squeeze()

        if idx_trial == 0:
            frs_filt = slice
        else:
            frs_filt = np.vstack((frs_filt, slice))
    return frs_filt





        # get firing rates for that trial between start and end time
        # copy the data
        # stack onto total firing rate data


def pca(num_pcs, frs, stationary=False):
    (num_bins, num_trials, num_neurons) = frs.shape
    frs_concat = np.swapaxes(frs, 0, 2).reshape((num_neurons, num_bins * num_trials))
    pca = PCA(n_components=num_pcs, svd_solver="full")
    pca.fit(frs_concat.T)
    fr_pcas = np.zeros((num_bins, num_trials, num_pcs))
    components = np.zeros((num_neurons, num_pcs))
    for j in range(num_pcs):
        component = pca.components_[j, :]
        components[:, j] = component
        fr_pcas[:, :, j] = np.tensordot(frs, component, axes=1)
    return fr_pcas, components, pca.explained_variance_


def filter_firing_rates_by_stim_type(frs, session, type='none'):
    """
    Select only firing rates whose trials corrspond to a given stimulation type:

    :param frs: firing rates (num_bins, num_trials, num_neurons)
    :param session:  Session instance
    :return: frs_filt (num_bins, num_trials_filtered, num_neurons) filtered data, trial_mask boolean numpy 1-vector
    """
    if type == 'none':

        stims = session.get_task_stimulation()
        mask = stims[:,0] == 0

    return frs[:, mask, :], mask