"""
spike_analysis Entry point for experimental data access & analysis.

"""
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import scripts
import utils




if __name__ == '__main__':
    session_data_dict = scripts.load_all_session_data(verbose=False)
    for sess in session_data_dict.values():
        try:
            scripts.epoch_based_pca(session_data_dict)
            # scripts.least_squares_prediction(sess, True)
        except KeyError:
            continue

        break
    # scripts.epoch_based_pca(session_data_dict)
    # num_pcs = 10
    # num_trials = 0
    # num_sessions = 0
    # lags = utils.ts_to_acorr_lags(list(session_data_dict.values())[0].get_ts())
    # ccrs = np.zeros((len(lags), num_pcs))
    # stds = np.zeros(ccrs.shape)
    #
    # for session in tqdm.tqdm(session_data_dict.values()):
    #     result = scripts.whitened_left_right_alm_crosscorr(session, num_pcs)
    #     if result:
    #         ccrs_i, stds_i, num_trials_i = result
    #         ccrs += ccrs_i
    #         stds += stds_i
    #         num_trials += num_trials_i
    #         num_sessions += 1
    #     # if num_sessions == 1:
    #     #     break
    #     #
    #
    # for idx_pc in range(num_pcs):
    #     plt.plot(lags, ccrs[:, idx_pc] / num_trials, label='i = {0}'.format(idx_pc+1))
    #     plt.fill_between(lags, ccrs[:, idx_pc] / num_trials - stds[:, idx_pc] / np.sqrt(num_trials), ccrs[:, idx_pc] / num_trials + stds[:, idx_pc] / np.sqrt(num_trials))
    #     plt.xlabel(r"Time Lag $\tau$")
    #     plt.ylabel(r"Cross-Correlation Coefficient")
    #     plt.title("Time-Lagged Cross-Correlation of Top {0} PC Pairs\n Left/Right ALM, N = {1} Trials, {2} Sessions".format(num_pcs, num_trials, num_sessions))
    # plt.xlim([-1, 1])
    # plt.legend()
    # plt.savefig("pc_crosscorr_58_sess.png", bbox_inches='tight', dpi=128)
    # plt.show()
    # exit(0)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

