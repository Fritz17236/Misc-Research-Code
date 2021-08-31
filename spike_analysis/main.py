"""
spike_analysis Entry point for experimental data access & analysis.

"""
import ctypes
from collections import defaultdict

import constants
import scripts
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import utils
import istarmap
import multiprocessing as mp
import os
import time
from sklearn.decomposition import PCA
import code


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
    significance_counts = scripts.trial_averaged_cch_significance_counts(spike_times, idx_trigger_neuron,
                                                                         idx_reference_neuron, rng, bin_width,
                                                                         window_width, p_crit, trial_len=constants.TIME_END_DEFAULT-constants.TIME_BEGIN_DEFAULT)
    significance_list.append(significance_counts)


if __name__ == '__main__':
    session_data_dict = scripts.load_all_session_data(verbose=False)
    # scripts.compute_experiment_pca(session_data_dict, overwrite=False)
    # scripts.pca_cross_correlation_experiment_averaged(session_data_dict, filter_lick_dirs=None)
    for sess_name, sess_data in session_data_dict.items():

        spike_times = sess_data.get_spike_times().copy()
        window_width = 6
        p_crit = constants.P_CRIT
        BIN_WIDTH = 1  # bin width in milliseconds
        RNG = 10  # number of bins to span in either direction of time
        num_trials = sess_data.get_num_trials()
        lags = BIN_WIDTH * np.arange(-RNG, RNG + 1)

        significance_counts = scripts.trial_averaged_cch_significance_counts(spike_times, 0, -10, RNG, BIN_WIDTH,
                                                                             window_width, p_crit, trial_len=constants.TIME_END_DEFAULT-constants.TIME_BEGIN_DEFAULT)
        lags = np.arange(-10,11)
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
            assert ((pair[1], pair[0]) not in neuron_pairs),\
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
            assert(idx in regions_to_neuron_indices[this_neuron_region])
            for region in region_list:
                if region != this_neuron_region:
                    assert(idx not in regions_to_neuron_indices[region])

        # check sum regions len = num neurons
        region_set_sizes = [len(regions_to_neuron_indices[reg]) for reg in region_list]
        assert(sum(region_set_sizes) == sess_data.get_num_neurons())

        spike_times = sess_data.get_spike_times().copy() # shape is [num_neurons][num_trials][variable num spikes]

        BIN_WIDTH = 1 # bin width in milliseconds
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
            assert((pair[1], pair[0]) not in neuron_pairs), "pair {0} was in neuron pairs but so was {1}".format(pair, (pair[1],pair[0]))
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

        np.save('means',cch_means)
        np.save('stds',cch_stds)
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
        #idx_pair, neuron_pairs, spike_times, BIN_WIDTH, mns, stds


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



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

