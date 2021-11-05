"""
scripts.py define and run multi-leveled analyses here

"""

# data I/O
import multiprocessing as mp
import traceback
from collections import defaultdict
import os

import statsmodels
from  statsmodels.tsa.stattools import adfuller, kpss

from tqdm import tqdm

import constants
from constants import DIR_RAW, DIR_SAVE
from utils import ts_to_acorr_lags, get_psd, get_whitening_filter, \
    autocorrelation, apply_filter, cross_correlation, \
    spike_train_cch_raw, filter_firing_rates_by_time, pca, filter_firing_rates_by_stim_type
import utils
import numpy as np
import matplotlib.pyplot as plt
from containers import Session
import tqdm
from sklearn.decomposition import PCA



# preprocess all data if need be, otherwise just loads the data handles into memory.


def load_all_session_data(verbose=False):
    """
    Load all session data, pre-processing (including PCA w/ 2 components) if necessary

    :return: session_data dictioanry object with keys being session names, and values being Session Instances.
    """

    def get_session_name_dict(file_name_list):
        """
        Organize a file_name list into sessions.

        Given a list of file names, organize them by session into a dictionary. The dictionary keys
        are the names of the session, and the dictionary values are a list of  names of files associated with that session.
        :param file_name_list:
        :return: session_name_dict
        """
        session_name_dict = defaultdict(list)
        for fname in file_name_list:
            if os.path.isdir(fname.rstrip()):
                continue
            split_name = fname.split('_')
            mat_header = split_name[0]
            session_name = split_name[1] + "_" + split_name[2] + "_" + split_name[3] + "_" + split_name[4]
            match_pattern = mat_header + "_" + session_name

            for file_name in file_name_list:
                if file_name.startswith(match_pattern) and file_name not in session_name_dict[session_name]:
                    session_name_dict[session_name].append(file_name)
        return session_name_dict

    print("Loading Session Data...")

    session_names = get_session_name_dict([f.name for f in os.scandir(DIR_RAW) if not f.is_dir()])
    session_data = {}

    for idx_session, (session_name, files) in enumerate(session_names.items()):
        if verbose:
            print("\nprocessing session {0}/{1}...".format(idx_session + 1, len(session_names.keys())))

        # create session instance to store data
        try:
            sess = Session(name=session_name, root=DIR_SAVE, verbose=verbose)
            for j, file_name in enumerate(files):
                if verbose:
                    print("\tadding file {0}/{1}".format(j + 1, len(files)))

                # create session object with given name, data saved in DIR_SAVE declared in constants.py
                sess.add_data(file_name=file_name, file_directory=DIR_RAW)

            # for accessing later in script
            session_data[session_name] = sess
        except Exception as e:
            with open(os.path.join(DIR_SAVE, "log.txt"), "w") as log:
                print("error loading session {0}".format(session_name))
                print("error loading session {0}".format(session_name), file=log)
                print(e)
                print(e, file=log)
                traceback.print_exc()
                traceback.print_exc(file=log)
                raise

    print("Session Data Loaded.")
    return session_data


# region Single-Session Scripts

def plot_pc_autocorrelation_analysis(session, pc_ref, pc_name, save_dir):
    """
    Plot the autocorrelation and power spectra of all prinicpal components in a given region

    :param pc_name:
    :param session: Session instance containing data
    :param ref_pc:  reference principal component used to decide whitening and autocorrelation
    :type session: Session
    """

    fs = 1 / (session.get_ts()[1] - session.get_ts()[0])
    lags = ts_to_acorr_lags(session.get_ts())

    for region in session.get_session_brain_regions():

        comps, projs = session.get_pca_by_region(region=region)
        num_bins, num_trials, num_pcs = projs.shape
        freqs, _ = get_psd(pc_ref[:, 0], fs)
        num_freqs = len(freqs)

        for idx_pc in range(num_pcs):
            autocorrs = np.zeros((num_bins, num_trials))
            autocorrs_whitened = np.zeros(autocorrs.shape)
            psds = np.zeros((num_freqs, num_trials))
            psds_whitened = np.zeros((num_freqs, num_trials))
            for idx_trial in range(num_trials):
                freqs, filt = get_whitening_filter(pc_ref[:, idx_trial], fs, b=np.inf, mode='highpass')
                try:
                    autocorrs[:, idx_trial] = autocorrelation(projs[:, idx_trial, idx_pc])
                except FloatingPointError:
                    continue
                autocorrs_whitened[:, idx_trial] = autocorrelation(apply_filter(projs[:, idx_trial, idx_pc], filt))
                freqs, psds[:, idx_trial] = get_psd(autocorrs[:, idx_trial], fs)
                _, psds_whitened[:, idx_trial] = get_psd(autocorrs_whitened[:, idx_trial], fs)

            title = "Autocorrelation_Region_{0}_PC_{1}_Whitened_wrt_{2}".format(region, idx_pc + 1, pc_name)
            save_path = os.path.join(save_dir, title + ".png")
            plt.figure(title)
            plt.title(title)
            error = np.std(autocorrs, axis=1) / np.sqrt(num_trials)
            y = np.mean(autocorrs, axis=1)
            error_whitened = np.std(autocorrs_whitened, axis=1) / np.sqrt(num_trials)
            y_whitened = np.mean(autocorrs_whitened, axis=1)
            plt.plot(lags, y, label="Unwhitened", c='r')
            plt.plot(lags, y_whitened, label='Whitened', c='g')
            plt.fill_between(lags, y_whitened - error_whitened, y_whitened + error_whitened, alpha=.2, color='r')
            plt.fill_between(lags, y_whitened - error_whitened, y_whitened + error_whitened, alpha=.2, color='g')
            plt.ylabel("Autocorrelation Coefficient")
            plt.xlabel(r"Time Lag $\tau$ (s)")
            plt.ylim([-1, 1])
            plt.legend()
            plt.savefig(save_path, dpi=128, bbox_inches='tight')

            title = "Power_Spectral_Density_Region_{0}_PC_{1}_Whitened_wrt_{2}".format(region, idx_pc + 1, pc_name)
            save_path = os.path.join(save_dir, title + ".png")

            plt.figure(title)
            plt.title(title)
            error = np.std(psds, axis=1) / np.sqrt(num_trials)
            y = np.mean(psds, axis=1)
            error_whitened = np.std(psds_whitened, axis=1) / np.sqrt(num_trials)
            y_whitened = np.mean(psds_whitened, axis=1)
            plt.loglog(freqs, y, label="Unwhitened", c='r')
            plt.loglog(freqs, y_whitened, label='Whitened', c='g')
            plt.fill_between(freqs, y - error, y + error, alpha=.2, color='r')
            plt.fill_between(freqs, y_whitened - error_whitened, y_whitened + error_whitened, alpha=.2, color='g')
            plt.ylabel("Power Spectral Density")
            plt.xlabel(r"Frequency (Hz)")
            plt.legend()
            plt.savefig(save_path, dpi=128, bbox_inches='tight')

    plt.close("all")


def plot_pc_cross_correlation_analysis(session, pc_ref, pc_name, save_dir):
    """
    Plot the autocorrelation and power spectra of all prinicpal components in a given region

    :param pc_name:
    :param session: Session instance containing data
    :param ref_pc:  reference principal component used to decide whitening and autocorrelation
    :type session: Session
    """

    fs = 1 / (session.get_ts()[1] - session.get_ts()[0])
    lags = ts_to_acorr_lags(session.get_ts())

    for region in session.get_session_brain_regions():

        comps, projs = session.get_pca_by_region(region=region)
        num_bins, num_trials, num_pcs = projs.shape
        freqs, _ = get_psd(pc_ref[:, 0], fs)
        num_freqs = len(freqs)

        for idx_pc in range(num_pcs):
            crosscorrs = np.zeros((num_bins, num_trials))
            crosscorrs_whitened = np.zeros(crosscorrs.shape)
            cross_psds = np.zeros((num_freqs, num_trials))
            cross_psds_whitened = np.zeros((num_freqs, num_trials))
            for idx_trial in range(num_trials):
                freqs, filt = get_whitening_filter(pc_ref[:, idx_trial], fs, b=np.inf, mode='highpass')
                try:
                    crosscorrs[:, idx_trial] = cross_correlation(pc_ref[:, idx_trial], projs[:, idx_trial, idx_pc])
                except FloatingPointError:
                    continue
                crosscorrs_whitened[:, idx_trial] = cross_correlation(apply_filter(pc_ref[:, idx_trial], filt)
                                                                      , apply_filter(projs[:, idx_trial, idx_pc], filt))
                freqs, cross_psds[:, idx_trial] = get_psd(crosscorrs[:, idx_trial], fs)
                _, cross_psds_whitened[:, idx_trial] = get_psd(crosscorrs_whitened[:, idx_trial], fs)

            title = r"Cross-Correlation_Region_{0}_PC_{1}_Whitened_wrt_{2}".format(region, idx_pc + 1, pc_name)
            save_path = os.path.join(save_dir, title + ".png")

            plt.figure(title)
            plt.title(title)
            error = np.std(crosscorrs, axis=1) / np.sqrt(num_trials)
            y = np.mean(crosscorrs, axis=1)
            error_whitened = np.std(crosscorrs_whitened, axis=1) / np.sqrt(num_trials)
            y_whitened = np.mean(crosscorrs_whitened, axis=1)
            plt.plot(lags, y, label="Unwhitened", c='r')
            plt.plot(lags, y_whitened, label='Whitened', c='g')
            plt.fill_between(lags, y_whitened - error_whitened, y_whitened + error_whitened, alpha=.2, color='r')
            plt.fill_between(lags, y_whitened - error_whitened, y_whitened + error_whitened, alpha=.2, color='g')
            plt.ylabel("Autocorrelation Coefficient")
            plt.xlabel(r"Time Lag $\tau$ (s)")
            plt.ylim([-1, 1])
            plt.legend()
            plt.savefig(save_path, dpi=128, bbox_inches='tight')

            title = r"Cross-Power_Spectral_Density_Region_{0}_PC_{1}_Whitened_wrt_{2}".format(region, idx_pc + 1,
                                                                                              pc_name)
            plt.figure(title)
            plt.title(title)
            error = np.std(cross_psds, axis=1) / np.sqrt(num_trials)
            y = np.mean(cross_psds, axis=1)
            error_whitened = np.std(cross_psds_whitened, axis=1) / np.sqrt(num_trials)
            y_whitened = np.mean(cross_psds_whitened, axis=1)
            plt.loglog(freqs, y, label="Unwhitened", c='r')
            plt.loglog(freqs, y_whitened, label='Whitened', c='g')
            plt.fill_between(freqs, y - error, y + error, alpha=.2, color='r')
            plt.fill_between(freqs, y_whitened - error_whitened, y_whitened + error_whitened, alpha=.2, color='g')
            plt.ylabel("Power Spectral Density")
            plt.xlabel(r"Frequency (Hz)")
            plt.legend()
            plt.savefig(os.path.join(save_dir, title + ".png"), dpi=128, bbox_inches='tight')

        plt.close("all")


def pairwise_pca_cross_correlation_trial_averaged(session, region_pair, pca_idxs, filter_lick_dirs='l'):
    """

    :param session: Session instance containing data
    :param region_pair: 2-tuple of strings specifying brain regions 1 and 2
    :param pca_idxs: 2-tuple of indices specifying which principal
    :param filter_lick_dirs: only consider trials where the task was a specific lick direction (left 'l' or right 'r')
    :return: (cross_corrs, cross_corrs_whitened, num_trials, n_trials): trial-averaged cross correlation between the
     specified pcs of the specified regions
    """
    if filter_lick_dirs:
        trial_mask = session.get_task_lick_directions() == filter_lick_dirs
    else:
        trial_mask = np.ones((session.get_num_trials(),), dtype=np.int)

    _, proj_1 = session.get_pca_by_region(region=region_pair[0])
    proj_1 = proj_1[:, trial_mask, pca_idxs[0]]
    _, proj_2 = session.get_pca_by_region(region=region_pair[1])
    proj_2 = proj_2[:, trial_mask, pca_idxs[1]]

    num_bins, num_trials = proj_1.shape

    assert (proj_1.shape == proj_2.shape), "Principal " \
                                           "components should have same shape but had {0} and {1}" \
                                           "".format(proj_1.shape, proj_2.shape)

    fs = 1 / (session.get_ts()[1] - session.get_ts()[0])
    freqs, _ = get_psd(proj_1[:, 0], fs)

    cross_corrs = np.zeros((num_bins, num_trials))
    cross_corrs_whitened = np.zeros(cross_corrs.shape)

    for idx_trial in range(num_trials):
        _, whiten_filter = get_whitening_filter(proj_1[:, idx_trial], fs, b=np.inf, mode='highpass')
        try:
            cross_corrs[:, idx_trial] = cross_correlation(proj_1[:, idx_trial], proj_2[:, idx_trial])
        except FloatingPointError:
            continue
        cross_corrs_whitened[:, idx_trial] = cross_correlation(apply_filter(proj_1[:, idx_trial], whiten_filter)
                                                               , apply_filter(proj_2[:, idx_trial], whiten_filter))

    return cross_corrs, cross_corrs_whitened, num_trials


def trial_cch_p_val_analysis(trig, ref, rng, bin_width, p_crit, cch_pred):
    cch = utils.spike_train_cch_raw(trig=trig, ref=ref, rng=rng, bin_width=bin_width)
    p_vals = utils.get_p_vals(cch, cch_pred)
    sigs = np.asarray([1 if p_vals[idx_prob] < p_crit else 0 for idx_prob in range(len(p_vals))])

    return {
        "cch": cch,
        "cch_pred": cch_pred,
        "p_vals": p_vals,
        "sigs": sigs
    }


def trial_averaged_cch_significance_counts(spike_times, idx_neuron_1, idx_neuron_2, rng, bin_width, window_width,
                                           p_crit, trial_len):

    cch_pred = utils.get_cch_predictor(spike_times[idx_neuron_1], spike_times[idx_neuron_2], rng, window_width, bin_width, trial_len)
    num_trials = spike_times.shape[1]
    lags = bin_width * np.arange(-rng, rng + 1)
    significance_counts = np.zeros(lags.shape)
    for idx_trial in range(num_trials):
        t_data = trial_cch_p_val_analysis(trig=spike_times[idx_neuron_1][idx_trial],
                                          ref=spike_times[idx_neuron_2][idx_trial], rng=rng, bin_width=bin_width,
                                          p_crit=p_crit, cch_pred=cch_pred)
        significance_counts += t_data["sigs"]
    return significance_counts

def trial_averaged_cch(session, idx_neuron_1, idx_neuron_2, bin_width, rng):
    """
    Compute the cch between two neurons averaging over all trials in a session

    :param session: Session instance
    :param idx_neuron_1: index of neuron 1 within session spike times
    :param idx_neuron_2: index of neuron 2 within session spike times
    :param bin_width: width of cch bins
    :param rng: number of bins to compute cch over
    :return: lags, cch, num_tirals (3-tuple 2 1d numpy arrays, 1 int)
    """
    num_trials = session.get_num_trials()

    spike_times = session.get_spike_times().copy()
    lags, cch = utils.spike_train_cch_raw(spike_times[idx_neuron_1][0], spike_times[idx_neuron_2][0], rng=rng,
                                          bin_width=bin_width, return_lags=True
                                          )
    cchs = np.zeros((len(cch), num_trials))
    cchs[:, 0] = cch
    print("Computing Trial-Averaged CCH...")
    for idx_trial in tqdm.tqdm(range(1, num_trials)):
        trig = spike_times[idx_neuron_1][idx_trial]
        ref = spike_times[idx_neuron_2][idx_trial]
        cchs[:, idx_trial] = spike_train_cch_raw(trig, ref, rng, bin_width)

    return lags, cchs.mean(axis=1), num_trials


def compute_session_pca_by_region(session, num_pcs=2, mode='fr', dt=constants.BIN_WIDTH_DEFAULT,
                                t_start=constants.TIME_BEGIN_DEFAULT, t_stop=constants.TIME_END_DEFAULT):
    """
    Perform Principal Component Analysis on stored Firing Rates, Splitting by brain region

    PCA is performed across trials
    :param num_pcs: number of principal components to compute.
    :return: region_pcs dict with keys = region (str) and values princpal components for that
    region, shape = [num_ts/bins][num_trials][num_pcs]
    """

    region_pcs ={}
    print("Computing Region PC's...")
    for idx, region in enumerate(session.get_session_brain_regions()):
        print("\tRegion: {0} ({1}/{2})".format(region, idx+1, len(session.get_session_brain_regions())))
        if mode == 'fr':
            frs = session.get_firing_rates(region=region).copy()
        elif mode == 'isi':
            frs = utils.compute_trial_isis(session, dt, t_start, t_stop, region)

        (num_bins, num_trials, num_neurons) = frs.shape
        frs_concat = np.swapaxes(frs, 0, 2).reshape((num_neurons, num_bins * num_trials))
        pca = PCA(n_components=num_pcs, svd_solver="full")
        try:
            pca.fit(frs_concat.T)
        except FloatingPointError:
            print("True divide by zero error encountered in pca for session {0}".format(session._name))
        fr_pcas = np.zeros((num_bins, num_trials, num_pcs))

        for j in range(num_pcs):
            component = pca.components_[j, :]
            fr_pcas[:, :, j] = np.tensordot(frs, component, axes=1)

        region_pcs[region] = fr_pcas
    return region_pcs

def compute_session_cch_predictors(session):
    # TODO: implement with main code, and add option to shuffle trials by given amount before returning
    pass


def pca_by_epoch(session: Session, region='left ALM', num_pcs = 5, diff=False, log=False):
    """
    Compute the pca for each epoch of session


    :param session: Session instance
    :param region: region to compute principal components over
    :param num_pcs: number of principal components to compute
    :return pcs_by_epoch: Dictionary whose keys are the epoch and whose values are principal component data
    """


    def pca_epoch(frs_filtered, frs_full):
        (num_bins, num_trials, num_neurons) = frs_full.shape
        pca = PCA(n_components=num_pcs, svd_solver="full")
        pca.fit(frs_filtered)
        components = np.zeros((num_neurons, num_pcs))
        fr_pcas = np.zeros((num_bins, num_trials, num_pcs))
        for j in range(num_pcs):
            component = pca.components_[j, :]
            components[:, j] = component
            fr_pcas[:, :, j] = np.tensordot(frs_full, component, axes=1)
        return fr_pcas, components, pca.explained_variance_

    def firing_rates_by_epoch(frs, session, epoch, trial_mask):
        epochs = ['sample', 'delay', 'response']

        if epoch == 'sample':
            data = session.get_task_sample_times()[:, trial_mask]

        elif epoch == 'delay':
            data = session.get_task_delay_times()[:, trial_mask]

        elif epoch == 'response':
            data = session.get_task_cue_times()[:, trial_mask]

        elif epoch == 'pre-sample':
            data = session.get_task_sample_times()[:, trial_mask]
            t_starts = np.zeros(data[0,:].shape)
            t_ends = data[0,:]
            return filter_firing_rates_by_time(frs,session.get_ts(), t_starts, t_ends)

        t_starts = data[0, :]
        t_ends = data[1, :] + data[0, :]

        return filter_firing_rates_by_time(frs,session.get_ts(), t_starts, t_ends)

    epochs = ['sample', 'delay', 'response', 'pre-sample']
    pcs_by_epoch = {}

    # firing rates, selecting only non-stimulation trials for PC comptutation
    frs, trial_mask = filter_firing_rates_by_stim_type(session.get_firing_rates(region=region), session)

    if diff:
        # make firing rates diff in time via first differnce
        frs[:-1, :, :] = np.diff(frs, axis=0)

    if log:
        frs = np.log(frs)
        frs[np.isnan(frs)] = 0
        frs[np.isinf(frs)] = 0


    # compute overall pca
    fr_pcas, components, spectrum = pca(num_pcs, frs)
    pcs_by_epoch['all'] = (fr_pcas, components, spectrum)

    for epoch in epochs:
        frs_filtered = firing_rates_by_epoch(frs, session, epoch, trial_mask)
        pcs_by_epoch[epoch] = pca_epoch(frs_filtered, frs)

    return pcs_by_epoch


def whitened_left_right_alm_crosscorr(session: Session, num_pcs = 5):
    """
    Compute the whitneed cross corrleation of each of [num_pcs] principal component pairs (across left/right hemisphere).

    :param session: session data containing left and right ALM activity
    :param num_pcs:  number of pcs to compute
    :return: ccrs, num_trials : cross correlation with size (num_bins) summed over num_trials.
    """
    # first check both left and right hemispheres are in this session, if not return none

    regions = session.get_session_brain_regions()
    if not (('left ALM' in regions) and ('right ALM' in regions)):
        return None
    else:
        # then select activity during the post-sample epoch (t > 0)
        ts = session.get_ts()
        start_idx = np.argwhere(ts > 0)[0][0]
        assert(np.all(ts[start_idx:] > 0))

        frs, trial_mask = filter_firing_rates_by_stim_type(session.get_firing_rates()[start_idx:, :, :], session)

        left_neurons = session.get_neuron_brain_regions() == 'left ALM'
        right_neurons = session.get_neuron_brain_regions() == 'right ALM'

        left_frs = frs[:, :, left_neurons]
        right_frs = frs[:, :, right_neurons]

        # compute the pcs, projections  of that activity for each hemisphere
        _, left_components, left_spectrum = pca(num_pcs, left_frs)
        _, right_components, right_spectrum = pca(num_pcs, right_frs)


        left_frs_full = session.get_firing_rates()[:, :, left_neurons][:, trial_mask, :]
        right_frs_full = session.get_firing_rates()[:, :, right_neurons][:, trial_mask, :]
        left_fr_pc = np.zeros((left_frs_full.shape[0], left_frs_full.shape[1], num_pcs))
        right_fr_pc = np.zeros((right_frs_full.shape[0], right_frs_full.shape[1], num_pcs))

        for j in range(num_pcs):
            left_fr_pc[:, :, j] = np.tensordot(left_frs_full, left_components[:, j], axes=1)
            right_fr_pc[:, :, j] = np.tensordot(right_frs_full, right_components[:, j], axes=1)

        lags = ts_to_acorr_lags(ts)
        num_trials = left_fr_pc.shape[1]

        ccrs = np.zeros((len(lags), num_trials, num_pcs))
        for idx_pc in range(num_pcs):
            for idx_trial in range(num_trials):
                left_traj = left_fr_pc[:, idx_trial, idx_pc]
                right_traj = right_fr_pc[:, idx_trial, idx_pc]
                ccrs[:, idx_trial, idx_pc] = utils.get_whitened_cross_correlation(x=left_traj, y=right_traj, fs=ts[3]-ts[2], bandwidth=np.inf, mode='highpass')

        ccrs[np.isnan(ccrs)] = 0
        return ccrs.sum(axis=1), ccrs.std(axis=1) / np.sqrt(num_trials),  num_trials
        # whiten both left and right trjaectories w.r.t. left trajectory
        # run cross-correlation of each add to running sums, num trials

    # return  sum, num trials
# endregion


# region Multi-Session Scripts

def get_all_session_brain_regions(session_data_dict):
    """
    Get a list of all brain regions present a set of sessions.

    :param session_data_dict: Dictionary with keys = session name, and value = Session instance.
    (output of load_all_session_data)
    :type session_data_dict: dict
    :return: brain_regions a list of brain regions contained within session data.

    """
    brain_regions_set = set()

    for sess_name, sess_data in session_data_dict.items():
        regions = sess_data.get_session_brain_regions()
        for r in regions:
            brain_regions_set.add(r)
    return brain_regions_set


def get_all_brain_region_pairs(brain_regions_set):
    """
    Get all pairwise combinations of a list of brain regions, excluding a region paired with itself.

    :param brain_regions_set:
    :return: brain_region_pairs list of pairs of brain regions (note pair(A, B) == pair(B,A) for two regions A,B.
    """
    brain_region_pairs = []

    for region_A in brain_regions_set:
        for region_B in brain_regions_set:
            if (region_A == region_B) or (region_B, region_A) in brain_region_pairs:
                continue
            else:
                brain_region_pairs.append((region_A, region_B))
    return brain_region_pairs


def get_session_to_region_dict(session_data_dict):
    """
    Get a dictionary mapping session name to regions in that session

    :param session_data_dict: session data dict with keys = session name and values = Session instance
    :return:
    """
    return {
        sess_name: sess_data.get_session_brain_regions()
        for sess_name, sess_data in session_data_dict.items()
    }


def match_region_pairs_to_sessions(session_to_regions):
    """
    Match each brain region pair to a list of sessions containing both of those regions.

    :param session_to_regions: dict with keys = sessname, values= regions in that session
    :return: region_pairs_to_session_names , dictionary mapping each pair of regions to session names.
    If a region pair has no sessions common (i.e its value is an empty list), then it is removed before returning
    :rtype: dict
    """
    # for each pair,
    # get all sessions that contain both region a and region b
    region_pairs_to_session_names = defaultdict(set)

    # get all sessions that have both regions
    for sess_name, regions in session_to_regions.items():
        # go through each session
        # get the regions of that session
        pairs = get_all_brain_region_pairs(regions)
        # get all the possible pairs of regions
        for region_A, region_B in pairs:
            for other_name, other_regions in session_to_regions.items():
                if other_name == sess_name:
                    continue
                elif region_A in other_regions and region_B in other_regions:
                    region_pairs_to_session_names[(region_A, region_B)].add(sess_name)

    # integrity check, and remove duplicates
    for pair, session_list in region_pairs_to_session_names.items():
        for sess_name in session_list:
            assert (pair[0] in session_to_regions[sess_name] and pair[1] in session_to_regions[sess_name])

        region_pairs_to_session_names[pair] = list(region_pairs_to_session_names[pair])

    return region_pairs_to_session_names


def pca_cross_correlation_experiment_averaged(session_data_dict, filter_lick_dirs='l'):
    """
    Compute the cross correlation between the first principal components of every brain region pair in experiment.

    :param session_data_dict: dict containing keys = session name, values = Session instances
    :param filter_lick_dirs: only consider trials where the task was a specific lick direction (left 'l' or right 'r')
    """

    def pairwise_pca_cross_correlation_experiment_averaged(pair, idx_pcs, region_pairs_to_sessions, session_data_dict,
                                                           filter_lick_dirs='l'):
        """
        Compute the weighted average of cross correlations between a pair pair of region principal components. The second
        region/pca is whitened with respect to the first (the first is used to compute whitenening filter.)

        :param pair: (2-tuple string) region names to check
        :param idx_pcs: (2-tuple int) indices specifying which principal component to compute from
        :param region_pairs_to_sessions: dict mapping region pairs to sessions containing both pairs
        :param session_data_dict: dict mapping session names to session data
        :param filter_lick_dirs: only consider trials where the task was a specific lick directrion
        (left 'l' or right 'r')
        :return: (cross_corrs, cross_corrs_whitened) pairwise correlation between region/pca,
        trial-averaged over all sessions
        """
        tot_trials = 0

        print("Averaging principal components over all sessions containing regions {0}".format(pair))

        for sess in tqdm.tqdm(region_pairs_to_sessions[pair]):
            try:
                cross_corrs, cross_corrs_whitened, nt = \
                    pairwise_pca_cross_correlation_trial_averaged(session_data_dict[sess], pair, pca_idxs=idx_pcs,
                                                                  filter_lick_dirs=filter_lick_dirs)
            except KeyError:
                print("Error computing cross-correlation on session {0} with brain regions {1}, components {2}".format(
                    sess, pair, idx_pcs
                ))
                raise

            cross_corr_stds = np.square(np.std(cross_corrs, axis=1))
            cross_corr_stds_whitened = np.square(np.std(cross_corrs_whitened, axis=1))  # easier to add variances
            cross_corrs = np.mean(cross_corrs, axis=1)
            cross_corrs_whitened = np.mean(cross_corrs_whitened, axis=1)
            if tot_trials == 0:
                cross_corr_stds_running = cross_corr_stds
                cross_corr_stds_whitened_running = cross_corr_stds_whitened
                cross_corrs_avg = cross_corrs
                cross_corrs_whitened_avg = cross_corrs_whitened
                tot_trials = nt
            else:
                cross_corr_stds_running += cross_corr_stds
                cross_corr_stds_whitened_running += cross_corr_stds_whitened

                cross_corrs_avg = (cross_corrs_avg * tot_trials + cross_corrs * nt) / (tot_trials + nt)
                cross_corrs_whitened_avg = (cross_corrs_whitened_avg * tot_trials +
                                            cross_corrs_whitened * nt) / (tot_trials + nt)
                tot_trials += nt

        return (
            cross_corrs_avg,
            cross_corrs_whitened_avg,
            np.sqrt(cross_corr_stds_running / tot_trials),
            np.sqrt(cross_corr_stds_whitened_running / tot_trials), tot_trials
        )

    session_to_regions = get_session_to_region_dict(session_data_dict)
    region_pairs_to_sessions = match_region_pairs_to_sessions(session_to_regions)
    # keep track of zero lags for each pair,
    # sort by strongest whitened
    zero_lags = defaultdict(tuple)
    for idx_pcs in [(0,0)]:
        for idx_pair, (pair, sess_list) in enumerate(region_pairs_to_sessions.items()):
            print("({2} / {3}) Pair: {0}, sessions containing pair: {1}".format(pair, len(sess_list), idx_pair + 1,
                                                                                len(region_pairs_to_sessions.items())))

            cc_pair, ccw_pair, std_pair, stdw_pair, num_trials = \
                pairwise_pca_cross_correlation_experiment_averaged(pair,
                                                                   idx_pcs=idx_pcs,
                                                                   region_pairs_to_sessions=region_pairs_to_sessions,
                                                                   session_data_dict=session_data_dict,
                                                                   filter_lick_dirs=filter_lick_dirs
                                                                   )

            p5 = 2 * num_trials ** -.5

            lags = utils.ts_to_acorr_lags(session_data_dict[sess_list[0]].get_ts())
            title = "Cross Correlation {0} & {1}, N={2}".format(pair[0], pair[1], num_trials)
            label = "Raw Data"
            utils.plot_data(lags, y=cc_pair, color="r", title=title,
                            ylim=[-1, 1], label=label, fill_error=3 * std_pair,
                            )

            if not os.path.exists(os.path.join(DIR_SAVE, "pairwise_ccs_experiment_averaged")):
                os.mkdir(os.path.join(DIR_SAVE, "pairwise_ccs_experiment_averaged"))

            pair_flat = (pair[0].replace(" ", "_"), pair[1].replace(" ", "_"))
            save_path = os.path.join(DIR_SAVE, "pairwise_ccs_experiment_averaged", "{0}_{1}_pc{2}_pc{3}.png".format(
                pair_flat[0], pair_flat[1], idx_pcs[0],
                idx_pcs[1]))
            label = "Whitened"
            utils.plot_data(lags, y=ccw_pair, color="g", save_path=save_path,
                            title=title, legend=True,
                            ylim=[-1, 1], label=label, fill_error=3 * stdw_pair,
                            vline=0, figsize=(16, 9)
                            )
            plt.close()
            zero_lags[pair] == ccw_pair[np.argwhere(np.isclose(lags, 0))]

    z_sorted = sorted(zero_lags.items(), key=lambda x: x[1], reverse=True)


def compute_experiment_pca(session_data_dict, **kwargs):
    for idx, (sess_name, data) in enumerate(session_data_dict.items()):
        print("Compute PC {0}/{1}".format(idx+1, len(session_data_dict.items())))
        data.compute_pca_by_brain_region(**kwargs)


def epoch_based_pca(session_data_dict):
    colors = {
        'all': 'k',
        'sample': 'r',
        'delay': 'g',
        'response': 'b',
        'pre-sample': 'y',
    }
    epochs = ['all', 'sample', 'delay', 'response', 'pre-sample']
    projs = defaultdict(list)
    # comps = defaultdict(list)
    spects = defaultdict(list)
    num_trials = 0
    for sess in session_data_dict.values():
        try:
            data = pca_by_epoch(sess, num_pcs=10, diff=True, log=True)
            epochs = [k for k in data.keys()]
            num_trials += data['all'][0].shape[1]
            adfs = {}
            kps = {}
            for e in epochs:
                fr_pcs, components, spectrum = data[e]
                projs[e].append(fr_pcs.sum(axis=1)[:, 0])  # get trial-averaged first PC
                # comps[e].append(components[:,0]) # get first PC


                plt.figure("comps")
                plt.plot(np.arange(components.shape[0]) + 1, components[:, 0], label=e, color=colors[e])
                plt.title("First Principal Component")
                plt.xlabel("Neuron Number")
                plt.ylabel("Weight")
                # adfs[e] = np.mean([adfuller(fr_pcs[:, idx_trial, 0])[1] for idx_trial in range(num_trials)])
                # kps[e] = np.mean([kpss(fr_pcs[:, idx_trial, 0], regression='ct')[1] for idx_trial in range(num_trials)])


                spects[e].append(spectrum / np.sum(spectrum))
        except KeyError:
            continue
        break

    plt.legend()
    plt.savefig("comps.png", bbox_inches='tight', dpi=128)

    num_sessions = len(projs['all'])
    num_bins = len(projs['all'][0])
    num_pcs = len(spects['all'][0])
    ts = sess.get_ts()
    for e in epochs:
        data_projs = np.zeros((num_sessions, num_bins))
        data_spects = np.zeros((num_sessions, num_pcs))
        # data_comps = np.zeros((num_sessions, num_neurons))
        for idx_sess in range(num_sessions):
            data_projs[idx_sess, :] = projs[e][idx_sess]
            data_spects[idx_sess, :] = spects[e][idx_sess]

        # plt.figure("projections")
        # plt.title("1st PC Projection, Average N = {0} Trials ({1} Sessions)".format(num_trials, num_sessions))
        # plt.ylabel("Projection of Activity onto PC1 (Unitless)")
        # plt.xlabel("Trial Time (s)")
        # mu = data_projs.sum(axis=0) / num_trials
        # sig = data_projs.std(axis=0) / np.sqrt(num_trials) * 0
        # ts = ts[:len(mu)]
        # plt.plot(ts, mu, label=e, c=colors[e])
        # plt.fill_between(ts, mu - sig, mu + sig, alpha=.2, color=colors[e])



        plt.figure("projections")
        plt.title("Differenced, Logged Firing Rates, Average N = {0} Trials ({1} Sessions)".format(num_trials, num_sessions))
        plt.ylabel("Projection of Activity onto PC1 (Unitless)")
        plt.xlabel("Trial Time (s)")
        mu = data_projs.sum(axis=0) / num_trials
        sig = data_projs.std(axis=0) / np.sqrt(num_trials) * 0
        ts = ts[:len(mu)]
        plt.plot(ts, mu, label=e, c=colors[e])
        adfs[e] = adfuller(mu)
        kps[e] = kpss(mu, regression='ct')



        plt.figure('spectra')
        plt.xlabel("PC Number")
        plt.ylabel("Fraction Variance Explained")
        mu = data_spects.mean(axis=0)
        sig = data_spects.std(axis=0) / np.sqrt(num_sessions) * 0
        plt.title(
            "PC Spectra (Explained Variance, Average N = {0} Trials, ({1} Sessions)".format(num_trials, num_sessions))
        plt.plot(np.arange(start=1, stop=num_pcs + 1), mu, label=e, c=colors[e])
        plt.fill_between(np.arange(start=1, stop=num_pcs + 1), mu - sig, mu + sig, alpha=.2, color=colors[e])





    plt.legend()
    plt.savefig("avg_spects.png", bbox_inches='tight', dpi=128)
    plt.figure("projections")
    plt.legend()
    plt.savefig("avg_projs.png", bbox_inches='tight', dpi=128)

    for e in adfs.keys():
        print("Epoch: {2}  - ADF: {0} - KPSS: {1}".format(adfs[e], kps[e], e))

    vvt = np.zeros(ts.shape)
    mvt = np.zeros(ts.shape)
    vvt2 = np.zeros(ts.shape)
    mvt2 = np.zeros(ts.shape)
    vvt3 = np.zeros(ts.shape)
    mvt3 = np.zeros(ts.shape)

    win_len = 5

    frs, trial_mask = filter_firing_rates_by_stim_type(sess.get_firing_rates(region='left ALM'), sess)
    print(frs.shape)

    frs_t = frs.mean(axis=(1,2))
    frs_t2 = (np.diff(frs, axis=0)).mean(axis=(1, 2))

    frs_t3 = np.log(frs)
    frs_t3[np.isnan(frs_t3)] = 0
    frs_t3[np.isinf(frs_t3)] = 0
    frs_t3 = (np.diff(frs_t3, axis=0)).mean(axis=(1, 2))
    for i in range(win_len, len(vvt)):
        vvt[i] = np.std(frs_t[i - win_len:i])
        vvt2[i] = np.std(frs_t2[i - win_len:i])
        vvt3[i] = np.std(frs_t3[i - win_len:i])

    dt = ts[1] - ts[0]
    win_size = dt * win_len

    plt.figure("varvt")
    plt.title("Sample Standard Deviation vs Time, Sliding Window of {0} ms".format(np.round(1000*win_size)))
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Value (Hz)")
    plt.plot(ts, vvt, label='Raw')
    plt.plot(ts, vvt2, label='Diff')
    plt.plot(ts, vvt3, label='Log-Diff')
    plt.legend()
    plt.savefig("varvt_trans.png", bbox_inches='tight', dpi=128)


    for i in range(win_len, len(vvt)):
        mvt2[i] = np.mean(frs_t2[i - win_len:i])
        mvt[i] = np.mean(frs_t[i - win_len:i])
        mvt3[i] = np.mean(frs_t3[i - win_len:i])
    dt = ts[1] - ts[0]
    win_size = dt * win_len

    # plt.plot(ts, mvt, label='Mean')

    plt.figure("meanvt")
    plt.title("Sample Mean vs Time, Sliding Window of {0} ms".format(np.round(1000 * win_size)))
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("Value (Hz)")
    plt.plot(ts, mvt, label='Raw')
    plt.plot(ts, mvt2, label='Diff')
    plt.plot(ts, mvt3, label='Log-Diff')
    plt.legend()
    plt.savefig("meanvt_trans.png",bbox_inches='tight', dpi=128)





    # plt.savefig("varvt.png", bbox_inches='tight', dpi=128)
    plt.show()


def least_squares_prediction(session, invert=False):
    def fit(X, b):
        from sklearn.model_selection import train_test_split
        X_train, X_test, b_train, b_test = train_test_split(X, b, test_size = 0.33, random_state = 42)
        z_star = np.linalg.pinv(X_train) @ b_train
        b_hat = X_test @ z_star
        b_hat[b_hat > 0] = 1
        b_hat[b_hat <= 0] = -1
        # J = 1 / len(b_hat) * np.sum([(b[i] - b_hat[i])**2 for i in range(len(b))])**.5
        correct = np.zeros(b_hat.shape)
        for i, dir in enumerate(b_hat):
            if b_hat[i] == b_test[i]:
                correct[i] = 1

        return np.mean(correct)

    transformations = ['raw', 'diff', 'log', 'diff-log', 'log-diff']
    frs_left, trial_mask = filter_firing_rates_by_stim_type(session.get_firing_rates(region='left ALM'), session)
    frs_right, _ = filter_firing_rates_by_stim_type(session.get_firing_rates(region='right ALM'), session)
    frs = np.concatenate((frs_left, frs_right), axis=2)
    b = session.get_task_lick_directions()[trial_mask]
    b = np.asarray([constants.ENUM_LICK_LEFT if d == 'l' else constants.ENUM_LICK_RIGHT for d in b])


    ts = session.get_ts()
    for t in transformations:
        if t == 'raw':
            frs_t = frs.copy()
            ls = 'o'
        elif t == 'diff':
            frs_t = frs.copy()
            frs_t[:-1] = np.diff(frs, axis=0)
            ls = '.'
            if invert:
                frs_t = np.cumsum(frs_t, axis=0)
        elif t == 'log':
            frs_t = np.log(frs)
            ls = '*'
            if invert:
                frs_t = np.exp(frs_t)
        elif t == 'diff-log':
            frs_t = frs.copy()
            frs_t[:-1] = np.diff(frs, axis=0)
            frs_t = np.log(frs_t)
            if invert:
                frs_t = np.exp(frs_t)
                frs_t = np.cumsum(frs_t, axis=0)
            ls='x'
        elif t == 'log-diff':
            frs_t = frs.copy()
            frs_t = np.log(frs_t)
            frs_t[:-1] = np.diff(frs, axis=0)
            ls = 'd'
            if invert:
                frs_t = np.cumsum(frs_t, axis=0)
                frs_t = np.exp(frs_t)
        else:
            continue
        frs_t[np.isnan(frs_t)] = 0
        frs_t[np.isinf(frs_t)] = 0
        fr_pcas, _, _ = pca(10, frs_t)
        js = [fit(fr_pcas[idx_time, :, :], b) for idx_time in tqdm.tqdm(range(len(ts)))]
        plt.figure("js")
        plt.plot(ts, js, label=str(t), marker=ls)

    plt.axhline(.5, c='black',label='Chance Level')
    plt.legend()
    plt.title("Lick Direction Prediction Accuracy")
    plt.xlabel("Trial Time")
    plt.ylabel("Prediction Accuracy")
    plt.savefig("fit_pc_projs_transformed.png", bbox_inches='tight', dpi=128)
    plt.show()

    # for each of (raw, diff, log, diff-log, log-diff)
        # apply transformation
        # get pcs
        # select a time point
        # compute pc projs
        # compute least squares predicition

def cch_analysis_multithreaded(sess_data):
    spike_times = sess_data.get_spike_times().copy()
    window_width = 6
    p_crit = constants.P_CRIT
    BIN_WIDTH = 1  # bin width in milliseconds
    RNG = 10  # number of bins to span in either direction of time
    num_trials = sess_data.get_num_trials()
    lags = BIN_WIDTH * np.arange(-RNG, RNG + 1)
    significance_counts = trial_averaged_cch_significance_counts(spike_times, 0, -10, RNG, BIN_WIDTH,
                                                                         window_width, p_crit,
                                                                         trial_len=constants.TIME_END_DEFAULT - constants.TIME_BEGIN_DEFAULT)
    lags = np.arange(-10, 11)
    plt.plot(lags, significance_counts)
    plt.show()
    ##########################################################
    # split neurons into region sets
    neuron_regions = sess_data.get_neuron_brain_regions()
    region_list = sess_data.get_session_brain_regions()
    regions_to_neuron_indices = defaultdict(list)
    for idx, region in enumerate(neuron_regions):
        regions_to_neuron_indices[region].append(idx)
    # check no neuron in more than one region
    for idx in range(len(neuron_regions)):
        this_neuron_region = neuron_regions[idx]
        assert (idx in regions_to_neuron_indices[this_neuron_region])
        for region in region_list:
            if region != this_neuron_region:
                assert (idx not in regions_to_neuron_indices[region])
    # check sum regions len = num neurons
    region_set_sizes = [len(regions_to_neuron_indices[reg]) for reg in region_list]
    assert (sum(region_set_sizes) == sess_data.get_num_neurons())
    # region  Get all pairs of neurons between the two regions
    idx_region_trig = 0
    idx_region_ref = 1
    neuron_pairs = []
    for idx_tri_nrn in np.arange(len(regions_to_neuron_indices[region_list[idx_region_trig]])):
        for idx_ref_nrn in np.arange(len(regions_to_neuron_indices[region_list[idx_region_ref]])):
            neuron_pairs.append((
                regions_to_neuron_indices[region_list[idx_region_trig]][idx_tri_nrn],
                regions_to_neuron_indices[region_list[idx_region_ref]][idx_ref_nrn]
            ))
    for pair in neuron_pairs:
        assert ((pair[1], pair[0]) not in neuron_pairs), \
            "pair {0} was in neuron pairs but so was {1}".format(pair, (pair[1], pair[0]))
    print("Neuron Pairs {0}: ".format(len(neuron_pairs)))
    # endregion
    num_pairs = len(neuron_pairs)
    return_list = mp.Manager().list()
    with mp.Pool() as pool:
        pool.starmap(bar, tqdm([
            (
                spike_times, neuron_pairs, idx_pair, RNG, BIN_WIDTH, window_width, p_crit, return_list
            )
            for idx_pair in range(num_pairs)],
            total=len(neuron_pairs)), chunksize=10)
    sig_counts = np.zeros((len(return_list), len(return_list[0])))
    np.save("sig_counts", sig_counts)
    significance_counts = sig_counts.sum(axis=0)
    num_trials = spike_times.shape[1]
    lags = BIN_WIDTH * np.arange(-RNG, RNG + 1)
    # plt.figure()
    # plt.scatter(lags, significance_counts)
    # plt.xlabel(r"Time lag $\tau$ (ms)")
    # plt.title("Significant Correlation Occurrences {3}-{4}, (p < {0}%, N ="
    #           " {1} trials, M = {2} Neurons)".format(constants.P_CRIT, num_trials, num_pairs,
    #                                                  region_list[idx_region_trig], region_list[idx_region_ref]))
    # plt.show()
    exit(0)
    #############################################################################
    num_trials = sess_data.get_num_trials()
    # split neurons into region sets
    neuron_regions = sess_data.get_neuron_brain_regions()
    region_list = sess_data.get_session_brain_regions()
    regions_to_neuron_indices = defaultdict(list)
    for idx, region in enumerate(neuron_regions):
        regions_to_neuron_indices[region].append(idx)
    # check no neuron in more than one region
    for idx in range(len(neuron_regions)):
        this_neuron_region = neuron_regions[idx]
        assert (idx in regions_to_neuron_indices[this_neuron_region])
        for region in region_list:
            if region != this_neuron_region:
                assert (idx not in regions_to_neuron_indices[region])
    # check sum regions len = num neurons
    region_set_sizes = [len(regions_to_neuron_indices[reg]) for reg in region_list]
    assert (sum(region_set_sizes) == sess_data.get_num_neurons())
    spike_times = sess_data.get_spike_times().copy()  # shape is [num_neurons][num_trials][variable num spikes]
    BIN_WIDTH = 1  # bin width in milliseconds
    RNG = 10  # number of bins to span in either direction of time
    # region  Get all pairs of neurons between the two regions
    idx_region_trig = 0
    idx_region_ref = 1
    neuron_pairs = []
    for idx_tri_nrn in np.arange(len(regions_to_neuron_indices[region_list[idx_region_trig]])):
        for idx_ref_nrn in np.arange(len(regions_to_neuron_indices[region_list[idx_region_ref]])):
            neuron_pairs.append((
                regions_to_neuron_indices[region_list[idx_region_trig]][idx_tri_nrn],
                regions_to_neuron_indices[region_list[idx_region_ref]][idx_ref_nrn]
            ))
    for pair in neuron_pairs:
        assert ((pair[1], pair[0]) not in neuron_pairs), "pair {0} was in neuron pairs but so was {1}".format(pair, (
        pair[1], pair[0]))
    print("Neuron Pairs {0}: ".format(len(neuron_pairs)))
    # endregion
    return_list = mp.Manager().list()
    with mp.Pool(processes=9) as pool:
        pool.starmap(foo, tqdm([
            (idx_pair, neuron_pairs, spike_times,
             num_trials,
             RNG, BIN_WIDTH, return_list
             ) for idx_pair in range(len(neuron_pairs))],
            total=len(neuron_pairs)),
                     chunksize=9
                     )
    print("return list has len {0}".format(len(return_list)))
    cch_means = np.zeros((2 * RNG + 1, len(neuron_pairs)))
    cch_stds = np.zeros(cch_means.shape)
    print(len(return_list[0]), return_list[0])
    for idx_pair, (mn, std) in enumerate(return_list):
        cch_means[:, idx_pair] = mn
        cch_stds[:, idx_pair] = std
    # cch_means = np.ctypeslib.as_array(cch_means_c)
    # cch_stds = np.ctypeslib.as_array(cch_stds_c)
    np.save('means', cch_means)
    np.save('stds', cch_stds)
    # with mp.Pool(processes=9) as pool:
    #     try:
    #         tqdm(
    #             pool.starmap(foo, [
    #                 (idx_pair, neuron_pairs, spike_times,
    #                  BIN_WIDTH, RNG, cch_means_c, cch_stds_c, num_trials
    #                  ) for idx_pair in range(len(neuron_pairs))]),
    #             total=len(neuron_pairs)
    #         )
    #     except Exception as e:
    #         print(e)
    #         raise
    # Get trigger and reference spike trains for a pair
    # idx_pair, neuron_pairs, spike_times, BIN_WIDTH, mns, stds
    # cch_means = np.ctypeslib.as_array(cch_means_c)
    # cch_stds = np.ctypeslib.as_array(cch_stds_c)
    #
    # np.save('means', cch_means)
    # np.save('stds', cch_stds)
    exit(0)
    # plt.figure(figsize=(16,9))
    # plt.title("CCH: {0} Trigger Neuron {1}, {2} Reference Neuron {3}, N = {4} Trials".format(
    #     region_list[idx_region_trig], idx_trigger_neuron, region_list[idx_region_ref], idx_reference_neuron, num_trials))
    # plt.xlabel(r"Time Lag $\tau$ (ms)")
    # plt.ylabel("Frequency")
    # plt.bar(lags, cchs.mean(axis=1))
    # # plt.errorbar(lags, cchs.mean(axis=1), yerr=cchs.std(axis=1)/np.sqrt(num_trials))
    # plt.show()
    # dt = .001
    # t_start = -1
    # t_stop = 0
    # ts = np.arange(t_start, t_stop, step=dt)
    # plt.ion()
    #
    #
    # isi_dict = {}
    # def do(region):
    #     #isis = np.power(utils.compute_trial_isis(sess_data, ts, region=region), -1)
    #     isis = utils.compute_trial_isis(sess_data, ts, region=region)
    #     isis[np.isnan(isis) | np.isinf(isis)] = 0
    #     isi_dict[region] = isis
    #     (num_bins, num_trials, num_neurons) = isis.shape
    #     isis_concat = np.swapaxes(isis, 0, 2).reshape((num_neurons, num_bins * num_trials))
    #     pca = PCA(n_components=2, svd_solver="full")
    #     pca.fit(isis_concat.T)
    #
    #     isi_pcas = np.zeros((num_bins, num_trials, 2))
    #     for j in [0, 1]:
    #         component = pca.components_[j, :]
    #         isi_pcas[:, :, j] = np.tensordot(isis, component, axes=1)
    #
    #     return isi_pcas
    #
    #
    #
    # isi_pcas_alm, isi_pcas_mid, isi_pcas_thal = \
    #     [do(region) for region in ['left ALM', 'left Midbrain', 'left Thalamus']]
    # lags = utils.ts_to_acorr_lags(ts)
    #
    # isi_alms = isi_dict['left ALM']
    # isi_mid = isi_dict['left Midbrain']
    # isi_thal = isi_dict['left Thalamus']
    #
    #
    # num_bins = len(ts)
    # num_trials = isi_pcas_alm.shape[1]
    # cross_corrs = np.zeros((num_bins, num_trials))
    # cross_corrs_whitened = np.zeros(cross_corrs.shape)
    # fs = 1 / dt
    # #
    # idx_pcs = (0,0)
    # for idx_trial in range(num_trials):
    #     _, whiten_filter = utils.get_whitening_filter(isi_pcas_alm[:, idx_trial, 0], fs, b=np.inf, mode='highpass')
    #     try:
    #         cross_corrs[:, idx_trial] = utils.cross_correlation(isi_pcas_alm[:, idx_trial, idx_pcs[0]], isi_pcas_thal[:, idx_trial, idx_pcs[1]])
    #     except FloatingPointError:
    #         continue
    #
    #     cross_corrs_whitened[:, idx_trial] = utils.cross_correlation(
    #         utils.apply_filter(isi_pcas_alm[:, idx_trial, idx_pcs[0]], whiten_filter),
    #         utils.apply_filter(isi_pcas_thal[:, idx_trial, idx_pcs[1]], whiten_filter))
    #
    # plt.ion()
    # lags = utils.ts_to_acorr_lags(ts)
    # lags_fr = utils.ts_to_acorr_lags(sess_data.get_ts())
    #
    # plt.figure(1)
    # plt.plot(lags, cross_corrs_whitened.mean(axis=1), label='ISI PC 1 Whitened')
    # plt.plot(lags, cross_corrs.mean(axis=1), label='ISI PC 1')
    # plt.legend()
    # plt.figure(2)
    # plt.title("Single Trial ISI vs Firing Rate 1st Principal Component")
    # plt.plot(ts, isi_pcas_thal[:,0,0],label="1/ISI")
    # pcs = sess_data.get_pca_by_region('left Thalamus')[1]
    # plt.plot(sess_data.get_ts(), pcs[:,0, 0], label = 'Binned Firing Rates')
    # plt.legend()
    # plt.figure(3)
    # plt.title('Single Trial ISI vs Firing Rate')
    # plt.plot(ts, isi_thal[:, 0, 0], label="1/ISI Left Thalamus")
    # frs = sess_data.get_firing_rates('left Thalamus')
    # plt.plot(sess_data.get_ts(), frs[:, 0, 0], label='Binned Firing Rates')
    #
    # plt.show()

# endregion


# region Other
def foo(idx_pair, neuron_pairs, spike_times, num_trials,  RNG, BIN_WIDTH, return_list):
    # region TRIAL AVERAGED CCH
    # run first trial to get lags and shape to initialize data

    idx_trigger_neuron = neuron_pairs[idx_pair][0]
    idx_reference_neuron = neuron_pairs[idx_pair][1]

    idx_trial = 0
    trig = spike_times[idx_trigger_neuron, idx_trial]
    ref = spike_times[idx_reference_neuron, idx_trial]


    cch = utils.spike_train_cch_raw(trig, ref, rng=RNG, bin_width=.001 * BIN_WIDTH)
    # intialize data and store results from first trial
    cchs = np.zeros((2 * RNG + 1, num_trials))
    cchs[:, 0] = cch

    # now run for the rest of the trials
    for idx_trial in range(1, num_trials):
        trig = spike_times[idx_trigger_neuron, idx_trial]
        ref = spike_times[idx_reference_neuron, idx_trial]
        cchs[:, idx_trial] = utils.spike_train_cch_raw(trig=trig, ref=ref, rng=RNG, bin_width=.001 * BIN_WIDTH)
    # endregion

    return_list.append((cchs.mean(axis=1), cchs.std(axis=1)))


def bar(spike_times, neuron_pairs, idx_pair, rng, bin_width, window_width, p_crit, significance_list):
    idx_trigger_neuron = neuron_pairs[idx_pair][0]
    idx_reference_neuron = neuron_pairs[idx_pair][1]
    significance_counts = trial_averaged_cch_significance_counts(spike_times, idx_trigger_neuron,
                                                                         idx_reference_neuron, rng, bin_width,
                                                                         window_width, p_crit, trial_len=constants.TIME_END_DEFAULT-constants.TIME_BEGIN_DEFAULT)
    significance_list.append(significance_counts)




# endregion
