"""
spike_analysis Entry point for experimental data access & analysis.

"""
import matplotlib.pyplot as plt
import numpy as np
import scripts
import utils
from sklearn.decomposition import PCA

from scripts import avg_pert

if __name__ == '__main__':
    # region load data and session select
    session_data_dict = scripts.load_all_session_data(verbose=False)
    sess_list = [s for s in session_data_dict.values()]
    sess_left_right_alm = [s for s in sess_list if ('left ALM' in s.get_session_brain_regions()) and ('right ALM' in s.get_session_brain_regions()) and avg_pert(s) <= -1.0]
    session = sess_left_right_alm[1]
    # endregion

    # region config sim and load sim-wide params
    stationary_invert = True
    stationary = True
    normalize = True
    num_pcs = 3
    stims = session.get_task_stimulation()
    dirs = session.get_task_lick_directions()
    cue_times = session.get_task_cue_times()
    ts = session.get_ts()
    if stationary:
        ts = ts[1:]
    dt = ts[1] - ts[0]
    # endregion

    # region non-perturbation trials, error estimation
    trial_mask_left_lick_no_stim = [idx for idx in range(session.get_num_trials())
                                    if (stims[idx, 0] == 0) and (dirs[idx] == 'l')]

    frs_left = session.get_firing_rates(region='left ALM')[:, trial_mask_left_lick_no_stim, :]
    frs_right = session.get_firing_rates(region='right ALM')[:, trial_mask_left_lick_no_stim, :]

    if stationary:
        frs_left = np.diff(frs_left, axis=0)
        frs_right = np.diff(frs_right, axis=0)

    num_bins, num_trials, n_l = frs_left.shape
    _, _, n_r = frs_right.shape
    frs_left_flat = frs_left.reshape((num_bins * num_trials, n_l))
    frs_right_flat = frs_right.reshape((num_bins * num_trials, n_r))
    frs_left_flat -= np.expand_dims(frs_left_flat.mean(axis=1), axis=1)
    frs_right_flat -= np.expand_dims(frs_right_flat.mean(axis=1),axis=1)

    A = np.hstack((frs_left_flat, -frs_right_flat))
    pca = PCA(n_components=num_pcs, svd_solver="full")
    pca.fit(A)
    err_pc = np.zeros((num_bins * num_trials, num_pcs))
    components = np.zeros((A.shape[1], num_pcs))
    for j in range(num_pcs):
        component = pca.components_[j, :]
        components[:, j] = component
        err_pc[:, j] = np.tensordot(A, component, axes=1)
    spec = pca.explained_variance_ratio_

    err_pc_r = err_pc.reshape((num_bins, num_trials, num_pcs))

    if stationary and stationary_invert:
        err_pc_r = np.cumsum(err_pc_r, axis=0)

    err_pc_r -= err_pc_r.mean(axis=0)
    std = err_pc_r.std(axis=1)
    for i in range(num_pcs):
        plt.figure('pc {0}'.format(i))
        plt.plot(ts, err_pc_r.mean(axis=1)[:,i]/std[:,i])
        for j in range(num_trials):
            plt.scatter(ts, err_pc_r[:, j, i]/std[:,i], c='r', alpha=.5, s=3)
            plt.axvline()
        plt.xlabel("Trial Time (s)")
        plt.ylabel("PC Projection (Normalized)")
        plt.title('Error Trajectories, {0} Trials'.format(num_trials))
        plt.savefig('pc {0}'.format(i), bbox_inches='tight', dpi=128)
    # endregion

    # region perturbation trials, error estimation
    trial_mask_left_lick_left_stim = [idx for idx in range(session.get_num_trials())
                                    if (stims[idx, 1] == 1) and (dirs[idx] == 'l')]
    stims_filt = stims[trial_mask_left_lick_left_stim, :]
    cue_times_filt = cue_times[:, trial_mask_left_lick_left_stim]
    t_perts = stims_filt[:, 2] - cue_times_filt[0,:]
    frs_left_p = session.get_firing_rates(region='left ALM')[:, trial_mask_left_lick_left_stim, :]
    frs_right_p = session.get_firing_rates(region='right ALM')[:, trial_mask_left_lick_left_stim, :]

    if stationary:
        frs_left_pt = np.diff(frs_left_p, axis=0)
        frs_right_pt = np.diff(frs_right_p, axis=0)
    else:
        frs_left_pt = frs_left_p
        frs_right_pt = frs_right_p

    _, num_trials_pert, _ = frs_left_pt.shape
    frs_left_flat_p = frs_left_pt.reshape((num_bins * num_trials_pert, n_l))
    frs_right_flat_p = frs_right_pt.reshape((num_bins * num_trials_pert, n_r))
    frs_left_flat_p -= np.expand_dims(frs_left_flat_p.mean(axis=1), axis=1)
    frs_right_flat_p -= np.expand_dims(frs_right_flat_p.mean(axis=1),axis=1)

    A_p = np.hstack((frs_left_flat_p, -frs_right_flat_p))

    err_pc_p = np.zeros((num_bins * num_trials_pert, num_pcs))
    for j in range(num_pcs):
        err_pc_p[:, j] = np.tensordot(A_p, components[:,j], axes=1)
    err_pc_r_p = err_pc_p.reshape((num_bins, num_trials_pert, num_pcs))
    if stationary and stationary_invert:
        err_pc_r_p = np.cumsum(err_pc_r_p, axis=0)

    err_pc_r_p -= np.expand_dims(err_pc_r.mean(axis=1),axis=1)
    err_pc_r_p -= err_pc_r_p.mean(axis=0)
    std_p = err_pc_r_p.std(axis=1)
    for i in range(num_pcs):
        plt.figure('pcp {0}'.format(i))
        plt.plot(ts, err_pc_r_p.mean(axis=1)[:,i]/std_p[:,i])
        for j in range(num_trials_pert):
            plt.scatter(ts, err_pc_r_p[:, j, i]/std_p[:,i], c='r', alpha=.5, s=3)
            plt.axvline(t_perts[j],c='k')
        plt.xlabel("Trial Time (s)")
        plt.ylabel("PC Projection (Normalized)")
        plt.title('Perturbed Error Trajectories, {0} Trials'.format(num_trials_pert))

        plt.savefig('pcp {0}'.format(i), bbox_inches='tight', dpi=128)
    # endregion

    print('done')


# region old code

    # _, s, vt = np.linalg.svd(A)
    # first = vt[0,:]
    #
    # err = A @ first
    #
    # Xl_flat = frs_left_flat @ first[:frs_left_flat.shape[1]]
    # Xr_flat = frs_right_flat @ first[frs_left_flat.shape[1]:]
    #
    # err_flat = A @ first
    # r_squared_flat = 1 - np.sum(err_flat**2, axis=0) / np.sum(Xr_flat**2, axis=0)
    # err = err_flat.reshape((num_bins, num_trials))
    #
    #
    # err = err.reshape((num_bins, num_trials))
    #
    # frs_left_p = session.get_firing_rates(region='left ALM')
    # trial_mask_left_lick_left_stim = [idx for idx in range(session.get_num_trials())
    #                                   if (stims[idx, 1] == 1)
    #                                   and
    #                                   (dirs[idx] == 'l')
    #                                   ]
    # frs_left_p = session.get_firing_rates(region='left ALM')[:, trial_mask_left_lick_left_stim, :]
    # frs_right_p = session.get_firing_rates(region='right ALM')[:, trial_mask_left_lick_left_stim, :]
    # _, num_trials_pert, _ = frs_left_p.shape
    # frs_left_flat_p = frs_left_p.reshape((num_bins * num_trials_pert, n_l))
    # frs_right_flat_p = frs_right_p.reshape((num_bins * num_trials_pert, n_r))
    # frs_left_flat_p = utils.z_score(frs_left_flat_p, axis=0)
    # frs_right_flat_p = utils.z_score(frs_right_flat_p, axis=0)
    # B = np.hstack((frs_left_flat_p, -frs_right_flat_p))
    # err_p = B @ first
    # err_p = err_p.reshape((num_bins, num_trials_pert))
    # std = err_p.std(axis=1)
    #
    # # realign based on perturbation time
    # # to eliminate the possibility that misaligned pert times are messing up the mean
    #
    # stims_filt = stims[trial_mask_left_lick_left_stim, :]
    # cue_times_filt = cue_times[:, trial_mask_left_lick_left_stim]
    #
    # t_perts = stims_filt[:, 2] - cue_times_filt[0,:]
    # idx_t_perts = [np.argwhere(ts >= tp)[0][0] for tp in t_perts]
    # width = 1.5 #second
    # idx_width = int(width / dt)
    # ts_rolled = np.arange(start=-idx_width, stop=idx_width) * dt
    # err_p_rolled = np.zeros((2*idx_width ,num_trials_pert))
    #
    # err_p_f = (err_p - np.expand_dims(err.mean(axis=1),axis=1))/np.expand_dims(std,axis=1)
    #
    # for idx_trial in range(err_p_rolled.shape[1]):
    #     itp = idx_t_perts[idx_trial]
    #     err_p_rolled[:, idx_trial] = err_p_f[itp-idx_width:itp+idx_width, idx_trial]
    #
    #
    # for idx_trial in range(err_p_rolled.shape[1]):
    #     plt.scatter(ts_rolled, err_p_rolled[:, idx_trial], c='r', alpha=.3)
    # plt.xlabel("Time Since Laser On")
    # plt.ylabel('Projection Difference')
    # plt.plot(ts_rolled, err_p_rolled.mean(axis=1))
    # plt.axvline(0, c='k')
    # plt.axvline(.5, c='k')

# endregion