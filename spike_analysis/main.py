"""
spike_analysis Entry point for experimental data access & analysis.

"""
import matplotlib.pyplot as plt
import numpy as np
import scripts
import utils
from sklearn.decomposition import PCA
import tqdm

from scripts import avg_pert


def plot_pc_projection(ts, projs, idx_pc, trials=True, mean=True, std_err=True, remove_outliers=True, append_title=''):
    """
    Plot Principal Component Projections.
    :param ts:
    :param projs:
    :param idx_pc:
    :param trials:
    :param mean:
    :param std_err:
    :param remove_outliers:
    :return:
    """
    num_bins, num_trials, num_pcs = projs.shape
    print(num_bins, num_trials, num_pcs)
    mu = projs.mean(axis=1)[:, idx_pc]
    sigma = projs.std(axis=1)[:, idx_pc] / np.sqrt(num_trials)

    plt.figure('pc ' + str(idx_pc) + append_title)
    plt.clf()
    if trials:
        for j in range(num_trials):
            plt.scatter(ts, projs[:, j, idx_pc], c='r', marker='.', alpha=.25, s=10)
    if mean:
        plt.plot(ts, projs.mean(axis=1)[:, idx_pc], c='b', linewidth=2)

    if std_err:
        plt.fill_between(ts, mu - sigma, mu + sigma, color='b', alpha=.3)

    plt.xlabel('Time (s)')
    plt.ylabel("PC Projection (Hz)")
    plt.show()

def plot_pc_weights(components, idx_pc, left_right_split=None):
    try:
        num_neurons, num_pcs = components.shape
    except ValueError:
        num_neurons = len(components)
        num_pcs = 1
        idx_pc = 0
        components = np.expand_dims(components, axis=1)

    plt.figure("pc weights {0}".format(idx_pc))
    plt.clf()

    plt.plot(np.arange(num_neurons), components[:,idx_pc], c='k')
    plt.scatter(np.arange(num_neurons), components[:, idx_pc], c='r',marker='.')
    plt.xlabel("Neuron Number")
    plt.ylabel("Projection Weight (Unitless)")
    plt.title("PC {0} Projection Weights".format(idx_pc))

    if left_right_split:
        plt.axvline(left_right_split, c='b')

    plt.show()

def filter_nan(x, fill=0):
    """
    Filter nan values to have fill instead
    :param x:
    :param fill:
    :return:
    """
    y = x.copy()
    y[np.isnan(y)] = fill
    return y

def nonnegative_pca(X, num_steps=1000):
    """
    Nonnegative PCA via http://web.stanford.edu/~montanar/RESEARCH/FILEPAP/nmf.pdf

    :param X: data matrix shape n x p
    :param num_steps: number of algorithm iterations
    :return: right,left pc vectors v, u respectively such that all are nonnegative
    """
    def l0_norm(x):
        return np.count_nonzero(x)

    def l2_norm(x):
        return np.sqrt(x.T @ x)

    def f(x):
        xp = pos(x)
        return filter_nan(np.sqrt(n) * xp / l2_norm(xp))


    def g(x):
        if np.isclose(l2_norm(x), 0):
            return 0 * x
        else:
            return filter_nan(np.sqrt(n) * x / l2_norm(x))

    def pos(x):
        y = x.copy()
        y[y < 0] = 0
        return y

    n,p = X.shape
    t = 0 # step

    u_tm1 = np.zeros((n,))
    u_t = np.zeros((n,))
    v_t = np.random.normal(size=(p,))
    v_t /= np.linalg.norm(v_t)
    v_tp1 = np.zeros((p,))

    us = [u_t]
    vs = [v_t]

    uhats = []
    vhats = [pos(v_t) / l2_norm(pos(v_t))]

    for t in tqdm.tqdm(range(num_steps)):
        b_t = l0_norm(pos(v_t)) / (np.sqrt(n) * l2_norm(pos(v_t)))
        u_t = X @ f(v_t) - b_t * g(u_tm1)
        d_t = np.sqrt(n) / l2_norm(u_t)

        v_tp1 = X.T @ g(u_t) - d_t * f(v_t)

        us.append(u_t)
        vs.append(v_t)
        uhats.append(u_t / l2_norm(u_t))
        vhats.append(pos(v_t) / l2_norm(pos(v_t)))

        u_tm1 = u_t.copy()
        v_t = v_tp1

    return uhats, vhats


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
    normalize = False
    num_pcs = 3
    stims = session.get_task_stimulation()
    dirs = session.get_task_lick_directions()
    cue_times = session.get_task_cue_times()
    sample_times = session.get_task_sample_times()
    ts = session.get_ts()
    if stationary:
        ts = ts[1:]
    dt = ts[1] - ts[0]
    # endregion

    # region non-perturbation trials, error estimation
    trial_mask_left_lick_no_stim = [idx for idx in range(session.get_num_trials())
                                    if (stims[idx, 0] == 0) and (dirs[idx] == 'l')]

    # trial_mask_left_lick_no_stim += [idx for idx in range(session.get_num_trials())
    #                                 if (stims[idx, 0] == 0) and (dirs[idx] == 'r')]


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

    if normalize:
        A = utils.z_score(A, axis=0)


    pca = PCA(svd_solver="full")
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


    idx_pc = 0

    std = err_pc_r.std(axis=1)
    # for i in range(num_pcs):
    #     plt.figure('pc {0}'.format(i))
    #     plt.clf()
    #     plt.plot(ts, np.abs(err_pc_r).mean(axis=1)[:,i]/std[:,i], linewidth=2)
    #     for j in range(num_trials):
    #         plt.scatter(ts, np.abs(err_pc_r)[:, j, i]/std[:,i], c='r', alpha=.3, s=3)
    #     plt.xlabel("Trial Time (s)")
    #     plt.ylabel("PC Projection (Normalized)")
    #     plt.title('Error Trajectories, {0} Trials'.format(num_trials))
    #     plt.savefig('pc {0}'.format(i), bbox_inches='tight', dpi=128)
    # endregion

    # region perturbation trials, error estimation
    trial_mask_left_lick_left_stim = [idx for idx in range(session.get_num_trials())
                                    if (stims[idx, 1] == 1) and (dirs[idx] == 'l')]
    # trial_mask_left_lick_left_stim += [idx for idx in range(session.get_num_trials())
    #                                 if (stims[idx, 1] == 1) and (dirs[idx] == 'r')]


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

    if normalize:
        A_p = utils.z_score(A_p, axis=0)
    err_pc_p = np.zeros((num_bins * num_trials_pert, num_pcs))
    for j in range(num_pcs):
        err_pc_p[:, j] = np.tensordot(A_p, components[:,j], axes=1)
    err_pc_r_p = err_pc_p.reshape((num_bins, num_trials_pert, num_pcs))
    if stationary and stationary_invert:
        err_pc_r_p = np.cumsum(err_pc_r_p, axis=0)

    err_pc_r_p -= np.expand_dims(err_pc_r.mean(axis=1),axis=1)
    err_pc_r_p -= err_pc_r_p.mean(axis=0)

    std_p = err_pc_r_p.std(axis=1)

    err_pruned = err_pc_r_p.copy()
    ep_mean = err_pruned.mean(axis=1)
    ep_std = err_pruned.std(axis=1)
    for i in range(num_pcs):
        for j in range(num_trials_pert):
            corr_mask = err_pruned[:, j, i] > 3 * ep_std[:,i]
            err_pruned[:, j, i][corr_mask] = ep_mean[:,i][corr_mask]


    # for i in range(num_pcs):
    #     plt.figure('pcp {0}'.format(i))
    #     plt.clf()
    #     plt.plot(ts, (err_pc_r_p).mean(axis=1)[:,i]/std_p[:,i], linewidth=2)
    #     for j in range(num_trials_pert):
    #         plt.scatter(ts, (err_pc_r_p[:, j, i])/std_p[:,i], c='r', alpha=.3, s=3)
    #         plt.axvline(t_perts[j],c='k')
    #         plt.axvline(t_perts[j]+.5, c='k')
    #     plt.xlabel("Trial Time (s)")
    #     plt.ylabel("PC {0} Error Projection (Normalized)".format(i))
    #     plt.title('PC {1} Perturbed Error Trajectories, {0} Trials'.format(num_trials_pert, i))
    #     plt.savefig('pcp {0}'.format(i), bbox_inches='tight', dpi=128)
    #
    #     plt.figure('pcp_abs {0}'.format(i))
    #     plt.clf()
    #     plt.plot(ts, np.abs(err_pc_r_p).mean(axis=1)[:,i]/std_p[:,i], linewidth=2)
    #     for j in range(num_trials_pert):
    #         plt.scatter(ts, np.abs(err_pc_r_p[:, j, i])/std_p[:,i], c='r', alpha=.3, s=3)
    #         plt.axvline(t_perts[j],c='k')
    #         plt.axvline(t_perts[j]+.5, c='k')
    #     plt.xlabel("Trial Time (s)")
    #     plt.ylabel("|PC {0} Error Projection| (Normalized)".format(i))
    #     plt.title('PC {1} |Perturbed Error| Trajectories, {0} Trials'.format(num_trials_pert, i))
    #     plt.savefig('pcp_abs {0}'.format(i), bbox_inches='tight', dpi=128)
    # endregion

    # region behavioral relevance
    trials_mask_all_left = trial_mask_left_lick_no_stim + trial_mask_left_lick_left_stim
    num_trials_left = len(trial_mask_left_lick_no_stim) + len(trial_mask_left_lick_left_stim)
    A_all = np.vstack((A, A_p))
    projs_all = A_all @ components
    projs_all = projs_all.reshape((num_bins, num_trials_left, num_pcs))

    if stationary and stationary_invert:
        projs_all = np.cumsum(projs_all, axis=0)

    projs_all -= np.mean(projs_all,axis=0)

    # get integrals for times after
    outcomes = session.get_behavior_report()[trials_mask_all_left]
    right_mask = outcomes == 1
    wrong_mask = outcomes == 0
    no_mask = outcomes == -1

    # outcomes[outcomes < 0] = 0
    t_samps = session.get_task_sample_times()[0,trials_mask_all_left]
    t_cues = session.get_task_cue_times()[0,trials_mask_all_left]

    idx_start = np.argwhere(ts>=(t_samps.mean()-cue_times[0,:].mean()))[0][0]
    idx_end = np.argwhere(ts >= 0)[0][0]

    projs_all -= np.mean(projs_all,axis=0)

    proj_cues = np.linalg.norm(projs_all, axis=0)

    proj_cues /= np.max(proj_cues,axis=0)
    print('done')


    for i in range(num_pcs):
        if i >= 3:
            break
        plt.figure('pc '+ str(i) + ' error2')
        plt.clf()
        plt.boxplot([proj_cues[right_mask, i], proj_cues[wrong_mask, i], proj_cues[no_mask, i]], sym='',meanline = True, showmeans = True)
        plt.scatter([1] * sum(right_mask), proj_cues[right_mask, i], c='k', marker='.', s=10)
        plt.scatter([2] * sum(wrong_mask), proj_cues[wrong_mask, i], c='k', marker='.', s=10)
        plt.scatter([3] * sum(no_mask), proj_cues[no_mask, i], c='k', marker='.', s=10)
        plt.axhline(0)
        plt.title("Norm of PC Trajectory, PC " + str(i))
        plt.ylabel("Projection Strength (Normalized)")
        plt.xticks([1, 2, 3], labels=['Correct Lick', 'Incorrect Lick', 'No Response'])
        plt.savefig('correctness_pc_{0}.png'.format(i), bbox_inches='tight', dpi=128)

    plt.figure('pc sum')
    plt.clf()
    data = np.mean(proj_cues,axis=1)
    plt.boxplot([data[right_mask], data[wrong_mask], data[no_mask]], sym='',meanline = True, showmeans = True)
    plt.scatter([1] * sum(right_mask), (proj_cues).mean(axis=1)[right_mask], c='k', marker='.', s=10)
    plt.scatter([2] * sum(wrong_mask), (proj_cues).mean(axis=1)[wrong_mask], c='k', marker='.', s=10)
    plt.scatter([3] * sum(no_mask), (proj_cues).mean(axis=1)[no_mask], c='k', marker='.', s=10)
    plt.axhline(0)
    plt.title("Projection Strength, Mean ")
    plt.ylabel("Projection Strength (Normalized)")
    plt.xticks([1, 2, 3], labels=['Correct Lick', 'Incorrect Lick', 'No Response'])
    plt.savefig('correctness_all_pcs.png',bbox_inches='tight', dpi=128)


    # endregion

    # region scratch
    idx_start = np.argwhere(ts>=(t_samps.mean()-cue_times[0,:].mean()))[0][0]
    idx_end = np.argwhere(ts >= 0)[0][0]
    proj_cues = np.linalg.norm(projs_all[idx_end:,:], axis=0)

    projs_all = (A_all @ components[:, 0]).reshape((num_bins, num_trials_left, 1))

    A_curr = A.copy()
    pcs = np.zeros((A.shape[1], num_pcs))
    for j in range(num_pcs):
        us,vs =  nonnegative_pca(A_curr)
        vm = vs[-1]
        pcs[:, j] = vm
        A_curr -= np.expand_dims(A_curr @ vm,axis=1) * np.expand_dims(vm,axis=1).T

    projs = (A @ components).reshape(num_bins,num_trials)
    projs_nn = (A @ pcs).reshape((num_bins, num_trials, num_pcs))

    if stationary and stationary_invert:
        projs = np.cumsum(projs,axis=0)
        projs_nn = np.cumsum(projs_nn,axis=0)


    demean=True
    if demean:
        projs -= np.mean(projs, axis=0)
        projs_nn -= np.mean(projs_nn, axis=0)


    proj_cues =  np.linalg.norm(projs[:,:,:], axis=0)
    proj_cues /= np.max(proj_cues, axis=0)
    proj_cues_nn = np.linalg.norm(projs_nn[:,:,:], axis=0)
    proj_cues_nn /= np.max(proj_cues_nn, axis=0)


    # endregion
    outcomes = session.get_behavior_report()[trial_mask_left_lick_no_stim]
    for i in range(num_pcs):
        right_mask = outcomes == 1
        wrong_mask = outcomes == 0
        no_mask = outcomes == -1

        # plt.figure('pc ' + str(i) + ' error')
        # plt.clf()
        # plt.boxplot([proj_cues[right_mask, i], proj_cues[wrong_mask, i], proj_cues[no_mask, i]], sym='', meanline=True,
        #             showmeans=True)
        # plt.scatter([1] * sum(right_mask), proj_cues[right_mask, i], c='k', marker='.', s=10)
        # plt.scatter([2] * sum(wrong_mask), proj_cues[wrong_mask, i], c='k', marker='.', s=10)
        # plt.scatter([3] * sum(no_mask), proj_cues[no_mask, i], c='k', marker='.', s=10)
        # plt.axhline(0)
        # plt.title("Norm of PC Trajectory, PC " + str(i))
        # plt.ylabel("Projection Strength (Normalized)")
        # plt.xticks([1, 2, 3], labels=['Correct Lick', 'Incorrect Lick', 'No Response'])


        plt.figure('pc ' + str(i) + ' error nn')
        plt.clf()
        plt.boxplot([proj_cues_nn[right_mask, i], proj_cues_nn[wrong_mask, i], proj_cues_nn[no_mask, i]], sym='', meanline=True,
                    showmeans=True)
        plt.scatter([1] * sum(right_mask), proj_cues_nn[right_mask, i], c='k', marker='.', s=10)
        plt.scatter([2] * sum(wrong_mask), proj_cues_nn[wrong_mask, i], c='k', marker='.', s=10)
        plt.scatter([3] * sum(no_mask), proj_cues_nn[no_mask, i], c='k', marker='.', s=10)
        plt.axhline(0)
        plt.title("Norm of PC Trajectory, PC " + str(i))
        plt.ylabel("Projection Strength (Normalized)")
        plt.xticks([1, 2, 3], labels=['Correct Lick', 'Incorrect Lick', 'No Response'])

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