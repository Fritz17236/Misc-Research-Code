"""
scripts.py define and run multi-leveled analyses here

"""

# data I/O
import traceback
from collections import defaultdict
import os
from time import time

import constants
from constants import DIR_RAW, DIR_SAVE
from utils import ts_to_acorr_lags, get_psd, get_whitening_filter, \
    autocorrelation, apply_filter, cross_correlation, \
    spike_train_cch_raw
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

# get all session regions (set)

# get all pairs of brain regions

# get names of all sessions that contain both pairs

# endregion
