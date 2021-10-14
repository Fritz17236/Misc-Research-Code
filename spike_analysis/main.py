"""
spike_analysis Entry point for experimental data access & analysis.

"""
from collections import defaultdict

import scripts
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    session_data_dict = scripts.load_all_session_data(verbose=False)
    colors = {
        'all': 'k',
        'sample': 'r',
        'delay': 'g',
        'response': 'b',
        'pre-sample': 'y',
    }
    epochs = ['all', 'sample', 'delay', 'response', 'pre-sample']
    projs  = defaultdict(list)
    # comps = defaultdict(list)
    spects = defaultdict(list)
    num_trials = 0
    for sess in session_data_dict.values():
        try:
            data = scripts.pca_by_epoch(sess, num_pcs=10)
            epochs = [k for k in data.keys()]
            num_trials += data['all'][0].shape[1]
            for e in epochs:
                fr_pcs, components, spectrum = data[e]
                projs[e].append(fr_pcs.sum(axis=1)[:, 0]) # get trial-averaged first PC
                # comps[e].append(components[:,0]) # get first PC
                plt.figure("comps")
                plt.plot(np.arange(components.shape[0])+1, components[:,0], label=e, color=colors[e])
                plt.title("First Principal Component")
                plt.xlabel("Neuron Number")
                plt.ylabel("Weight")

                spects[e].append(spectrum / np.sum(spectrum))
        except KeyError:
            continue
        break
    plt.legend()
    plt.savefig("comps.png", bbox_inches='tight', dpi=128)

    plt.show()

    num_sessions = len(projs['all'])
    num_bins = len(projs['all'][0])
    num_pcs = len(spects['all'][0])
    ts =  sess.get_ts()
    for e in epochs:
        data_projs = np.zeros((num_sessions, num_bins))
        data_spects = np.zeros((num_sessions, num_pcs))
        # data_comps = np.zeros((num_sessions, num_neurons))
        for idx_sess in range(num_sessions):
            data_projs[idx_sess, :] = projs[e][idx_sess]
            data_spects[idx_sess, :] = spects[e][idx_sess]

        plt.figure("projections")
        plt.title("1st PC Projection, Average N = {0} Trials ({1} Sessions)".format(num_trials, num_sessions))
        plt.ylabel("Projection of Activity onto PC1 (Unitless)")
        plt.xlabel("Trial Time (s)")
        mu = data_projs.sum(axis=0) / num_trials
        sig = data_projs.std(axis=0)  / np.sqrt(num_trials) * 0
        plt.plot(ts,  mu, label=e, c=colors[e])
        plt.fill_between(ts, mu - sig, mu + sig, alpha=.2, color=colors[e])

        plt.figure('spectra')
        plt.xlabel("PC Number")
        plt.ylabel("Fraction Variance Explained")
        mu = data_spects.mean(axis=0)
        sig = data_spects.std(axis=0) / np.sqrt(num_sessions) *0
        plt.title("PC Spectra (Explained Variance, Average N = {0} Trials, ({1} Sessions)".format(num_trials, num_sessions))
        plt.plot(np.arange(start=1, stop=num_pcs+1), mu, label=e, c=colors[e])
        plt.fill_between(np.arange(start=1, stop=num_pcs+1), mu - sig, mu + sig, alpha=.2, color=colors[e])

    plt.legend()
    plt.savefig("avg_spects.png", bbox_inches='tight', dpi=128)
    plt.figure("projections")
    plt.legend()
    plt.savefig("avg_projs.png",bbox_inches='tight', dpi=128)

    plt.show()












# See PyCharm help at https://www.jetbrains.com/help/pycharm/

