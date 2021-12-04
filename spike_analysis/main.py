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


def plot_pc_weights(components, idx_pc, spect, left_right_split=None, avg=False):
    try:
        num_neurons, num_pcs = components.shape
    except ValueError:
        num_neurons = len(components)
        num_pcs = 1
        idx_pc = 0
        components = np.expand_dims(components, axis=1)

    pct = np.round(spect[idx_pc] * 100, decimals=2)
    plt.figure("pc weights {0}".format(idx_pc))
    plt.clf()
    plt.plot(np.arange(num_neurons), components[:,idx_pc], c='k')
    plt.scatter(np.arange(num_neurons), components[:, idx_pc], c='r',marker='.')
    plt.xlabel("Neuron Number")
    plt.ylabel("Projection Weight (Unitless)")
    plt.title("PC {0} Projection Weights \n  Variance Explained: {1}%".format(idx_pc, pct))
    # plt.ylim([0, 1])

    if left_right_split:
        plt.axvline(left_right_split, c='b')
        if avg:
            left_avg = np.mean(components[:left_right_split,idx_pc])
            right_avg =  np.mean(components[left_right_split:,idx_pc])
            plt.axhline(left_avg, c='g',label='left hemisphere average = {0}'.format(left_avg))
            plt.axhline(right_avg, c='b',label='right hemisphere average = {0}'.format(right_avg))
            plt.legend()
    plt.savefig('pc_{0}_weights.png'.format(idx_pc), bbox_inches='tight',dpi=128)


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


def nn_pca(X, num_pcs=3, num_steps=100):
    """
    Nonnegative PCA
    :param X: 2d numpy array (num_observations x num_varaibles)
    :param num_pcs: (int number of principal components to compute)
    :param num_steps: (int numbner of steps to iterate when computing each pc
    :return: pcs, 2d numpy array (num_variables x  num_pcs)
    """

    def nn_pca_helper(X, num_steps=num_steps):
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

        n, p = X.shape
        t = 0  # step

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

    X_curr = X.copy()
    pcs = np.zeros((X.shape[1], num_pcs))


    var_samp = np.var(X_curr, axis=0)
    var_tot = np.sum(var_samp)
    for j in range(num_pcs):
        us, vs = nn_pca_helper(X_curr)
        vm = vs[-1]
        pcs[:, j] = vm
        X_curr -= np.expand_dims(X_curr @ vm, axis=1) * np.expand_dims(vm, axis=1).T

    projs = X @ pcs
    spect = np.var(projs, axis=0) / var_tot
    # sort by variance explained
    indices = np.flip(np.argsort(spect))
    return pcs[:, indices], spect[indices]


def error_pca(session, num_pcs, pcs=None, lick_direction='l', perturbation='l', stationary=True, stationary_invert=True, z_score_rates_before_pca=True, demean_projections=True, epoch='all', mode='error', equal_weight_hemi=True, nn=False):
    """
    Run error PCA analysis on a session
    :param session: Session instance containing data
    :param num_pcs: number of principal components to decompose
    :param pcs: option to seed principal components to avoid recomputation if already known
    :param lick_direction: 'l', 'r' or 'both': lick directions of data to analyze
    :param perturbation: 'neither', 'l', 'r' or 'both': perturbations of hemispheres of data to analyze
    :param stationary: bool: perform stationarity (differencing) transformation on data before pca analysis
    :param stationary_invert: invert stationarity transformation for resulting pc projections (cumsum)
    :param z_score_rates_before_pca: zero-mean and unit-std firing rates across time before flattening and pca
    :param demean_projections: zero mean pca projections across time after pca analysis
    :return: tuple with:
        projs, 3d numpy array (num_bins, num_trials, num_pcs)
        pcs 2d numpy array (num_neurons, num_pcs)
        trial_mask subset of session.get_num_trial() where the given filters (hemisphere, etc) are true, with num_trials true entries
    """
    stims = session.get_task_stimulation()
    dirs = session.get_task_lick_directions()


    if perturbation == 'l':
        stim_val = 1
    elif perturbation == 'r':
        stim_val = 2
    elif perturbation == 'both':
        stim_val = 6
    elif perturbation == 'neither':
        stim_val = 0
    else:
        raise Exception("bad arg passed for perturbation: " + str(perturbation))

    if lick_direction == 'l':
        dir_val = 'l'
    elif lick_direction == 'r':
        dir_val = 'r'
    elif lick_direction == 'both':
        dir_val = 'both'


    trial_mask = [idx for idx in range(session.get_num_trials())
                  if
                  ((stims[idx, 1] == stim_val) or (stims[idx,0] == stim_val))
                  and
                  ((dirs[idx] == dir_val) or dir_val == 'both')
                  ]

    ts = session.get_ts()
    if epoch == 'all':
        frs_left = session.get_firing_rates(region='left ALM')[:, trial_mask, :]
        frs_right = session.get_firing_rates(region='right ALM')[:, trial_mask, :]
    elif epoch == 'pre-cue':
        idx_stop = np.argwhere(ts > 0)[0][0]
        frs_left = session.get_firing_rates(region='left ALM')[:idx_stop, trial_mask, :]
        frs_right = session.get_firing_rates(region='right ALM')[:idx_stop, trial_mask, :]


    if z_score_rates_before_pca:
        frs_left -= frs_left.mean(axis=0)
        frs_right -= frs_right.mean(axis=0)
        # stds_left = frs_left.std(axis=0)
        # stds_right = frs_right.std(axis=0)
        # for i in range(frs_left.shape[1]):
        #     for j in range(frs_left.shape[2]):
        #         if stds_left[i,j] > 0:
        #             frs_left[:,i,j] /= stds_left[i,j]
        #     for j in range(frs_right.shape[2]):
        #         if stds_right[i,j] > 0:
        #             frs_right[:,i,j] /= stds_right[i,j]

    if stationary:
        ts = ts[1:]
        frs_left = np.diff(frs_left, axis=0)
        frs_right = np.diff(frs_right, axis=0)




    num_bins, num_trials, n_l = frs_left.shape
    _, _, n_r = frs_right.shape

    if mode == 'error':
        data = np.hstack((frs_left.reshape((num_bins * num_trials, n_l)), -frs_right.reshape((num_bins * num_trials, n_r))))
    elif mode =='redundant':
        data = .5 * np.hstack((frs_left.reshape((num_bins * num_trials, n_l)), frs_right.reshape((num_bins * num_trials, n_r))))
    else:
        raise Exception("bad argument {0} passed to mode in error pca".format(mode))

    if equal_weight_hemi:
        data[:, :n_l] /= (n_l)
        data[:, n_l:] /= (n_r)

        # data[:, :n_l] /= (data[:,:n_l].std(axis=0) * n_l)
        # data[:, n_l:] /= (data[:,n_l:].std(axis=0) * n_r)


    if np.all(pcs == None):
        if nn:
            pcs, spect = nn_pca(data, num_pcs)
        else:
            pca = PCA(svd_solver="full")
            pca.fit(data)
            pcs = np.zeros((data.shape[1], num_pcs))
            for j in range(num_pcs):
                pcs[:, j] = pca.components_[j,:]
            spect =  pca.explained_variance_ratio_
    else:
        spect = None

    projs = (data @ pcs).reshape((num_bins, num_trials, num_pcs))

    if stationary_invert:
        projs = np.cumsum(projs, axis=0)

    if demean_projections:
        projs -= projs.mean(axis=0)

    return projs, pcs, trial_mask, n_l, n_r, spect


# # # scratch
# for j in range(num_pcs):
#     plot_pc_weights(pcs, j, spect, n_l)
#
# # projs = proj_to_std_vs_t(projs)
#
# for j in range(num_pcs):
#     plt.figure("pc {0}".format(j))
#     plt.clf()
#     outcomes = session.get_behavior_report()[trial_mask]
#     right_mask = outcomes == 1
#     wrong_mask = outcomes == 0
#     no_mask = outcomes == -1
#     for i in range(projs.shape[1]):
#         if right_mask[i]:
#             color = 'g'
#             marker = 'o'
#         elif wrong_mask[i]:
#             color = 'maroon'
#             marker = 'x'
#         elif no_mask[i]:
#             color = 'grey'
#             marker = 'x'
#         plt.scatter(ts, projs[:, i, j], c=color, alpha=.3, marker=marker, s=20)
#     plt.plot(ts, projs[:, right_mask, :].mean(axis=1)[:, j], c='lime', label='correct', markeredgecolor='k')
#     plt.plot(ts, projs[:, wrong_mask, :].mean(axis=1)[:, j], c='r', label='incorrect', markeredgecolor='k')
#     plt.plot(ts, projs[:, no_mask, :].mean(axis=1)[:, j], c='k', label='no response', markeredgecolor='k')
#     plt.xlabel("Time (s)")
#     plt.ylabel(r"Standard Deviation (Hz)")
#     plt.title('PC {0} Trial Standard Deviation vs. \n Time, Lick Left, No Perturbations \n{1} Trials'.format(j, projs.shape[1]))
#     plt.legend()


def behavior_plot(projs, outcomes, normalize=True, title=None, savename=None):

    norms = projs.std(axis=0)
    if normalize:
        norms /= np.nanmax(norms, axis=0)

    right_mask = outcomes == 1
    wrong_mask = outcomes == 0
    no_mask = outcomes == -1

    # do scatter plot of all points on top of b&w
    # no symbols in b&w plot
    # xlabel = trial type
    # ylabel = standard deviation normalized
    # title : trial outcomes vs trial stnadard deviation along pc {0}
    # sort pcs by average std separation? separate function to get indices
    for j in range(projs.shape[2]):
        plt.figure('pc {0}'.format(j))
        plt.clf()
        plt.boxplot([norms[right_mask, j], norms[wrong_mask, j], norms[no_mask, j]], sym='', showmeans=True,  meanline=True)
        plt.scatter([1] * sum(right_mask), norms[right_mask, j], c='k', marker='.')
        plt.scatter([2] * sum(wrong_mask), norms[wrong_mask, j], c='k', marker='.')
        plt.scatter([3] * sum(no_mask), norms[no_mask, j], c='k', marker='.')
        plt.xticks(ticks=[1, 2, 3], labels=['Correct', 'Incorrect', 'No Repsonse'])
        plt.ylabel('Trial Standard Deviation \n (Normalized)')
        plt.xlabel('\n Trial Outcome')
        if title:
            plt.title('PC {0} '.format(j) + title)
        if savename:
            plt.savefig(savename + '_pc_{0}.png'.format(j), bbox_inches='tight', dpi=128)



    # norms = norms.mean(axis=1)
    # plt.figure('pc mean'.format(j))
    # plt.clf()
    # plt.boxplot([norms[right_mask], norms[wrong_mask], norms[no_mask]], sym='', showmeans=True, meanline=True)
    # plt.scatter([1] * sum(right_mask), norms[right_mask], c='k', marker='.')
    # plt.scatter([2] * sum(wrong_mask), norms[wrong_mask], c='k', marker='.')
    # plt.scatter([3] * sum(no_mask), norms[no_mask], c='k', marker='.')
    # plt.xticks(ticks=[1, 2, 3], labels=['Correct', 'Incorrect', 'No Repsonse'])
    # plt.ylabel('PC-Averaged Trial Standard Deviation')
    # plt.xlabel('Trial Outcome')


    # for each pc:
        # first do box and whisker plot for all 3
        # then do scatter plot for each
        # then format etc.

    # scatter plot each point for 3 columns
    # then do box & whisker plot


def proj_to_std_vs_t(projs, normalize=True):
    num_bins, num_trials, num_pcs = projs.shape
    stds = np.zeros(projs.shape)
    for j in range(num_pcs):
        for i in range(num_trials):
            for k in range(3, num_bins ):
                stds[k, i, j] = np.std(projs[:k, i, j])

    return stds


def plot_single_session(session):
    ts = session.get_ts()
    if stationary:
        ts = ts[1:]

    projs, pcs, trial_mask, n_l, n_r, spect = error_pca(session, num_pcs, None, lick_direction, perturbation, stationary,
                                                        stationary_invert, z_score_rates_before_pca, demean_projections,
                                                        epoch, mode)

    for j in range(num_pcs):
        plot_pc_weights(pcs, j, spect, left_right_split=n_l)


    projs_ll_pl, _, trial_mask_ll_pl, n_l, n_r, _ = error_pca(session, num_pcs, pcs, lick_direction='l', perturbation='l')
    projs_ll_pr, _, trial_mask_ll_pr, n_l, n_r, _ = error_pca(session, num_pcs, pcs, lick_direction='l', perturbation='r')
    projs_ll_pb, _, trial_mask_ll_pb, n_l, n_r, _ = error_pca(session, num_pcs, pcs, lick_direction='l', perturbation='both')

    projs_lr_np, _, trial_mask_lr_np, n_l, n_r, _ = error_pca(session, num_pcs, pcs, lick_direction='r', perturbation='neither')
    projs_lr_pl, _, trial_mask_lr_pl, n_l, n_r, _ = error_pca(session, num_pcs, pcs, lick_direction='r', perturbation='l')
    projs_lr_pr, _, trial_mask_lr_pr, n_l, n_r, _ = error_pca(session, num_pcs, pcs, lick_direction='r', perturbation='r')
    projs_lr_pb, _, trial_mask_lr_pb, n_l, n_r, _ = error_pca(session, num_pcs, pcs, lick_direction='r', perturbation='both')

    projs = proj_to_std_vs_t(projs)
    projs_ll_pl = proj_to_std_vs_t(projs_ll_pl)
    projs_ll_pr = proj_to_std_vs_t(projs_ll_pr)
    projs_ll_pb = proj_to_std_vs_t(projs_ll_pb)

    projs_lr_np = proj_to_std_vs_t(projs_lr_np)
    projs_lr_pl = proj_to_std_vs_t(projs_lr_pl)
    projs_lr_pr = proj_to_std_vs_t(projs_lr_pr)
    projs_lr_pb = proj_to_std_vs_t(projs_lr_pb)

    # region plot pc lick left no pert trials
    for j in range(num_pcs):
        plt.figure("pc {0}".format(j))
        plt.clf()
        outcomes = session.get_behavior_report()[trial_mask]
        right_mask = outcomes == 1
        wrong_mask = outcomes == 0
        no_mask = outcomes == -1
        for i in range(projs.shape[1]):
            if right_mask[i]:
                color = 'g'
                marker = 'o'
            elif wrong_mask[i]:
                color = 'maroon'
                marker = 'x'
            elif no_mask[i]:
                color = 'grey'
                marker = 'x'
            plt.scatter(ts, projs[:, i, j], c=color, alpha=.3, marker=marker, s=20)
        plt.plot(ts, projs[:, right_mask, :].mean(axis=1)[:, j], c='lime', label='correct', markeredgecolor='k')
        plt.plot(ts, projs[:, wrong_mask, :].mean(axis=1)[:, j], c='r', label='incorrect', markeredgecolor='k')
        plt.plot(ts, projs[:, no_mask, :].mean(axis=1)[:, j], c='k', label='no response', markeredgecolor='k')
        plt.xlabel("Time (s)")
        plt.ylabel(r"Standard Deviation (Hz)")
        plt.title('PC {0} Trial Standard Deviation vs. \n Time, Lick Left, No Perturbations \n{1} Trials'.format(j, projs.shape[1]))
        plt.legend()
        plt.savefig('ss_pc_{0}_lick_left_no_pert.png'.format(j),bbox_inches='tight', dpi=128)
    # endregion

    # region plot pc lick left left pert trials
    for j in range(num_pcs):
        plt.figure("pc {0}".format(j))
        plt.clf()
        outcomes = session.get_behavior_report()[trial_mask_ll_pl]
        right_mask = outcomes == 1
        wrong_mask = outcomes == 0
        no_mask = outcomes == -1
        for i in range(projs_ll_pl.shape[1]):
            if right_mask[i]:
                color='g'
                marker='o'
            elif wrong_mask[i]:
                color='maroon'
                marker='x'
            elif no_mask[i]:
                color='grey'
                marker='x'
            plt.scatter(ts, projs_ll_pl[:, i, j], c=color, marker=marker, alpha=.3, s=20)
        plt.plot(ts, projs_ll_pl[:,right_mask,:].mean(axis=1)[:,j],c='lime',label='correct',markeredgecolor='k')
        plt.plot(ts, projs_ll_pl[:, wrong_mask, :].mean(axis=1)[:, j], c='r', label='incorrect',markeredgecolor='k')
        plt.plot(ts, projs_ll_pl[:, no_mask, :].mean(axis=1)[:, j], c='k', label='no response',markeredgecolor='k')
        plt.axvline(-1.2, c='k')
        plt.axvline(-.7, c='k')
        plt.xlabel("Time (s)")
        plt.ylabel(r"Standard Deviation $\sigma(t)$ (Hz)")
        plt.title('PC {0} Projection Standard Deviation vs. Time, Lick Left, Left Perturbations {1} Trials'.format(j,projs_ll_pl.shape[1]))
        plt.legend()
        plt.savefig('ss_pc_{0}_lick_left_left_pert.png'.format(j),bbox_inches='tight', dpi=128)
    # endregion

    # region plot pc lick left right pert trials
    for j in range(num_pcs):
        plt.figure("pc {0}".format(j))
        plt.clf()
        outcomes = session.get_behavior_report()[trial_mask_ll_pr]
        right_mask = outcomes == 1
        wrong_mask = outcomes == 0
        no_mask = outcomes == -1
        for i in range(projs_ll_pr.shape[1]):
            if right_mask[i]:
                color='g'
                marker='o'
            elif wrong_mask[i]:
                color='maroon'
                marker='x'
            elif no_mask[i]:
                color='grey'
                marker='x'
            plt.scatter(ts, projs_ll_pr[:, i, j], c=color, marker=marker, alpha=.3, s=20)
        plt.plot(ts, projs_ll_pr[:,right_mask,:].mean(axis=1)[:,j],c='lime',label='correct', markeredgecolor='k', markeredgewidth=2)
        plt.plot(ts, projs_ll_pr[:, wrong_mask, :].mean(axis=1)[:, j], c='r', label='incorrect',markeredgecolor='k', markeredgewidth=2)
        plt.plot(ts, projs_ll_pr[:, no_mask, :].mean(axis=1)[:, j], c='k', label='no response',markeredgecolor='k', markeredgewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel(r"Standard Deviation $\sigma(t)$ (Hz)")
        plt.title('PC {0} Projection Standard Deviation vs. Time, Lick Left, Right Perturbations {1} Trials'.format(j,projs_ll_pr.shape[1]))
        plt.axvline(-1.2, c='k')
        plt.axvline(-.7, c='k')
        plt.legend()
        plt.savefig('ss_pc_{0}_lick_left_right_pert.png'.format(j),bbox_inches='tight', dpi=128)
    # endregion

    # region plot pc lick left both pert trials
    for j in range(num_pcs):
        plt.figure("pc {0}".format(j))
        plt.clf()
        outcomes = session.get_behavior_report()[trial_mask_ll_pb]
        right_mask = outcomes == 1
        wrong_mask = outcomes == 0
        no_mask = outcomes == -1
        for i in range(projs_ll_pb.shape[1]):
            if right_mask[i]:
                color='g'
                marker='o'
            elif wrong_mask[i]:
                color='maroon'
                marker='x'
            elif no_mask[i]:
                color='grey'
                marker='x'
            plt.scatter(ts, projs_ll_pb[:, i, j],marker=marker, c=color, alpha=.3, s=20)
        plt.plot(ts, projs_ll_pb[:,right_mask,:].mean(axis=1)[:,j],c='lime',label='correct')
        plt.plot(ts, projs_ll_pb[:, wrong_mask, :].mean(axis=1)[:, j], c='r', label='incorrect')
        plt.plot(ts, projs_ll_pb[:, no_mask, :].mean(axis=1)[:, j], c='k', label='no response')
        plt.axvline(-1.2, c='k')
        plt.axvline(-.7, c='k')
        plt.xlabel("Time (s)")
        plt.ylabel(r"Standard Deviation $\sigma(t)$ (Hz)")
        plt.title('PC {0} Projection Standard Deviation vs. Time, Lick Left, Both Perturbations {1} Trials'.format(j,projs_ll_pb.shape[1]))
        plt.legend()
        plt.savefig('ss_pc_{0}_lick_left_both_pert.png'.format(j),bbox_inches='tight', dpi=128)
    # endregion

    # region plot pc lick right no pert trials
    for j in range(num_pcs):
        plt.figure("pc {0}".format(j))
        plt.clf()
        outcomes = session.get_behavior_report()[trial_mask_lr_np]
        right_mask = outcomes == 1
        wrong_mask = outcomes == 0
        no_mask = outcomes == -1
        for i in range(projs_lr_np.shape[1]):
            if right_mask[i]:
                color='g'
                marker='o'
            elif wrong_mask[i]:
                color='maroon'
                marker='x'
            elif no_mask[i]:
                color='grey'
                marker='x'
            plt.scatter(ts, projs_lr_np[:, i, j],marker=marker, c=color, alpha=.3, s=20)
        plt.plot(ts, projs_lr_np[:,right_mask,:].mean(axis=1)[:,j],c='lime',label='correct')
        plt.plot(ts, projs_lr_np[:, wrong_mask, :].mean(axis=1)[:, j], c='r', label='incorrect')
        plt.plot(ts, projs_lr_np[:, no_mask, :].mean(axis=1)[:, j], c='k', label='no response')
        plt.xlabel("Time (s)")
        plt.ylabel(r"Standard Deviation $\sigma(t)$ (Hz)")
        plt.title('PC {0} Trial Standard Deviation vs. Time, \n Lick Right, No Perturbations \n {1} Trials'.format(j,projs_lr_np.shape[1]))
        plt.legend()
        plt.savefig('ss_pc_{0}_lick_right_no_pert.png'.format(j),bbox_inches='tight', dpi=128)
    # endregion

    # region plot pc lick right left pert trials
    for j in range(num_pcs):
        plt.figure("pc {0}".format(j))
        plt.clf()
        outcomes = session.get_behavior_report()[trial_mask_lr_pl]
        right_mask = outcomes == 1
        wrong_mask = outcomes == 0
        no_mask = outcomes == -1
        for i in range(projs_lr_pl.shape[1]):
            if right_mask[i]:
                color='g'
                marker='o'
            elif wrong_mask[i]:
                color='maroon'
                marker='x'
            elif no_mask[i]:
                color='grey'
                marker='x'
            plt.scatter(ts, projs_lr_pl[:, i, j], marker=marker, c=color, alpha=.3, s=20)
        plt.plot(ts, projs_lr_pl[:,right_mask,:].mean(axis=1)[:,j],c='lime',label='correct')
        plt.plot(ts, projs_lr_pl[:, wrong_mask, :].mean(axis=1)[:, j], c='r', label='incorrect')
        plt.plot(ts, projs_lr_pl[:, no_mask, :].mean(axis=1)[:, j], c='k', label='no response')
        plt.xlabel("Time (s)")
        plt.ylabel(r"Standard Deviation $\sigma(t)$ (Hz)")
        plt.title('PC {0} Projection Standard Deviation vs. Time, Lick Right, Left Perturbations {1} Trials'.format(j,projs_lr_pl.shape[1]))
        plt.legend()
        plt.savefig('ss_pc_{0}_lick_right_left_pert.png'.format(j),bbox_inches='tight', dpi=128)
    # endregion

    # region plot pc lick right right pert trials
    for j in range(num_pcs):
        plt.figure("pc {0}".format(j))
        plt.clf()
        outcomes = session.get_behavior_report()[trial_mask_lr_pr]
        right_mask = outcomes == 1
        wrong_mask = outcomes == 0
        no_mask = outcomes == -1
        for i in range(projs_lr_pr.shape[1]):
            if right_mask[i]:
                color='g'
                marker='o'
            elif wrong_mask[i]:
                color='maroon'
                marker='x'
            elif no_mask[i]:
                color='grey'
                marker='x'
            plt.scatter(ts, projs_lr_pr[:, i, j], c=color, marker=marker, alpha=.3, s=20)
        plt.plot(ts, projs_lr_pr[:,right_mask,:].mean(axis=1)[:,j],c='lime',label='correct')
        plt.plot(ts, projs_lr_pr[:, wrong_mask, :].mean(axis=1)[:, j], c='r', label='incorrect')
        plt.plot(ts, projs_lr_pr[:, no_mask, :].mean(axis=1)[:, j], c='k', label='no response')
        plt.xlabel("Time (s)")
        plt.ylabel(r"Standard Deviation $\sigma(t)$ (Hz)")
        plt.title('PC {0} Projection Standard Deviation vs. Time, Lick Right, Right  Perturbations {1} Trials'.format(j,projs_lr_pr.shape[1]))
        plt.legend()
        plt.savefig('ss_pc_{0}_lick_right_right_pert.png'.format(j),bbox_inches='tight', dpi=128)
    # endregion

    # region plot pc lick right both pert trials
    for j in range(num_pcs):
        plt.figure("pc {0}".format(j))
        plt.clf()
        outcomes = session.get_behavior_report()[trial_mask_lr_pb]
        right_mask = outcomes == 1
        wrong_mask = outcomes == 0
        no_mask = outcomes == -1
        for i in range(projs_lr_pb.shape[1]):
            if right_mask[i]:
                color='g'
                marker='o'
            elif wrong_mask[i]:
                color='maroon'
                marker='x'
            elif no_mask[i]:
                color='grey'
                marker='x'
            plt.scatter(ts, projs_lr_pb[:, i, j], c=color, marker=marker, alpha=.3, s=20)
        plt.plot(ts, projs_lr_pb[:,right_mask,:].mean(axis=1)[:,j],c='lime',label='correct')
        plt.plot(ts, projs_lr_pb[:, wrong_mask, :].mean(axis=1)[:, j], c='r', label='incorrect')
        plt.plot(ts, projs_lr_pb[:, no_mask, :].mean(axis=1)[:, j], c='k', label='no response')
        plt.xlabel("Time (s)")
        plt.ylabel(r"Standard Deviation $\sigma(t)$ (Hz)")
        plt.title('PC {0} Projection Standard Deviation vs. Time, Lick Right, Both Perturbations {1} Trials'.format(j,projs_lr_pb.shape[1]))
        plt.legend()
        plt.savefig('ss_pc_{0}_lick_right_both_pert.png'.format(j),bbox_inches='tight', dpi=128)
    # endregion


def get_mean_separability(projs, outcomes, ts, normalize=True, sort=False, stop=1.5):
    """
    Sort trials by outcomes, compute separability as difference of variability means, and return average variability
    of each pc for each trial, sorted by separabiilty
    :param projs: (num_bins, num_trials, num_pcs)
    :param outcomes:
    :return: avgs (3 x num_pcs) average variability of each of three ooutocomes (correct, incorrect, none) for each PC in projs, sorted by separability, num_trials
    avgs[0, j] gives avg correct variability (std) of trials along pc j
    avgs[1, j] '      ' incorrect ' '
    avgs[2, j] '      ' no        ' '
    """
    _, num_trials, num_pcs = projs.shape
    right_mask = outcomes == 1
    wrong_mask = outcomes == 0
    no_mask = outcomes == -1

    stop_idx = np.argwhere(ts > stop)[0][0]
    norms = projs[:stop_idx,:,:].std(axis=0)

    if normalize:
        norms /= np.nanmax(norms, axis=0)

    avgs = np.zeros((3, num_pcs))
    for j in range(num_pcs):
        avgs[0, j] = norms[right_mask, j].mean(axis=0)
        avgs[1, j] = norms[wrong_mask, j].mean(axis=0)
        avgs[2, j] = norms[no_mask, j].mean(axis=0)


    # sort avgs by separability between correct and no response trials
    if sort:
        seps = avgs[0,:] - avgs[2,:]
        indices = np.flip(np.argsort(seps))
        return avgs[:, indices], num_trials
    else:
        return avgs, num_trials


def get_pc_coms(pcs, n_l):
    """
    Get the "center of mass" of pc projection weights between left and right hemisphere
    :param pcs: pc weights (num pcs x num_neurons)
    :param n_l: left/right split (int)
    :return: coms (num_pcs,) centers of mass
    """
    coms = np.zeros((num_pcs,))
    for j in range(num_pcs):
        coms[j] = ((-1) * pcs[:n_l, j]).sum() + ((1) * pcs[n_l:, j]).sum()
    return coms

if __name__ == '__main__':
    # region load data and session select
    session_data_dict = scripts.load_all_session_data(verbose=False)
    sess_list = [s for s in session_data_dict.values()]
    sess_left_right_alm = [s for s in sess_list if ('left ALM' in s.get_session_brain_regions()) and ('right ALM' in s.get_session_brain_regions()) and avg_pert(s) <= -1.0]

    num_pcs = 3
    stationary = True
    stationary_invert = True
    z_score_rates_before_pca = True
    demean_projections = True
    lick_direction = 'l'
    perturbation = 'neither'
    epoch = 'all'
    mode = 'error'
    equal_weight_hemi = True
    nn = True

    num_sessions = len(sess_left_right_alm)
    avgs = np.zeros((3, num_sessions))
    num_trials = np.zeros((num_sessions,))
    pcs_coms = np.zeros((num_pcs, num_sessions))
    # endregion
    for idx_sess, session in tqdm.tqdm(enumerate(sess_left_right_alm)):
        pcs = None
        projs, pcs, trial_mask, n_l, n_r, spect = error_pca(session, num_pcs, pcs, lick_direction, perturbation, stationary, stationary_invert, z_score_rates_before_pca, demean_projections,epoch, mode, equal_weight_hemi, nn)
        outcomes = session.get_behavior_report()[trial_mask]
        pcs_coms[:, idx_sess] = get_pc_coms(pcs, n_l)
        ts = session.get_ts()[:projs.shape[0]]
        avg_sess, num_trials_sess = get_mean_separability(projs, outcomes, ts, normalize=True)
        if lick_direction == 'l':
            idx_max = np.flip(np.argsort(pcs_coms[:,idx_sess], axis=0))[0]
        elif lick_direction == 'r':
            idx_max = np.argsort(pcs_coms[:,idx_sess], axis=0)[0]

        avgs[:, idx_sess] = avg_sess[:,idx_max]

        num_trials[idx_sess] = num_trials_sess

    plt.figure('sess_avg')
    plt.clf()
    avgs_filt = avgs.copy()
    last = avgs_filt[2,:]
    mask = ~np.isnan(last)
    last = last[mask]
    plt.boxplot([avgs_filt[0, :], avgs_filt[1, :], last], sym='', showmeans=True,  meanline=True)
    for j in range(num_sessions):
        plt.scatter([1, 2, 3], avgs_filt[:, j], marker='.', c='k', alpha=.5)
    # for j in range(3):
    #     if j == 2:
    #         weighted = (last.T @ num_trials[mask]) / num_trials[mask].sum()
    #         plt.axhline(weighted, c='k', label='no response trial-weighted average  {0}'.format(j))
    #     elif j == 1:
    #         weighted = (avgs[j, :].T @ num_trials) / num_trials.sum()
    #         plt.axhline(weighted, c='r', label='incorrect response trial-weighted average  {0}'.format(j))
    #     else:
    #         weighted = (avgs[j, :].T @ num_trials) / num_trials.sum()
    #         plt.axhline(weighted, c='g', label='correct response trial-weighted average  {0}'.format(j))

    mu = np.nanmean(avgs_filt, axis=1)
    sig = np.nanstd(avgs_filt, axis=1)  / np.sqrt(num_sessions)
    plt.errorbar([1, 2, 3], mu, yerr =sig, c='g')
    plt.plot([1, 2, 3], np.nanmean(avgs_filt, axis=1), c='g', marker='x')
    plt.ylim([0, 1])
    plt.ylabel("Trial-Averaged Standard Deviation")
    plt.xlabel('\n Trial Outcome')
    plt.legend()
    plt.xticks(ticks=[1, 2, 3], labels=['Correct', 'Incorrect', 'No Response'])
    plt.title('Trial-Averaged Standard Deviation vs Outcome  \n Along Most Separable PC (of {1}) \n Lick Left, No Perturbation Trials \n {0} Sessions'.format(num_sessions, num_pcs))

    plt.savefig("all_sess_sep.png",bbox_inches='tight',dpi=128)
    print('done')
