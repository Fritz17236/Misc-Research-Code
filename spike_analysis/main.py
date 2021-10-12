"""
spike_analysis Entry point for experimental data access & analysis.

"""

import scripts
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
    projs  = {
        e: [] for e in epochs
    }
    spects = {
        e: [] for e in epochs
    }
    for sess in session_data_dict.values():
        try:
            data = scripts.pca_by_epoch(sess, num_pcs=10)

            epochs = [k for k in data.keys()]
            for e in epochs:
                fr_pcs, components, spectrum = data[e]

                plt.figure("Projections")
                plt.plot(sess.get_ts(), fr_pcs.mean(axis=1)[:,0], label=e, c=colors[e])
                projs[e].append(fr_pcs.mean(axis=1)[:, 0])

                plt.figure('spectra')
                plt.plot(spectrum, label=e, c=colors[e])
                spects[e].append(spectrum)

            plt.figure('Projections')

            plt.figure('spectra')
        except KeyError:
            continue

        exit(0)

    num_sessions = len(projs)










# See PyCharm help at https://www.jetbrains.com/help/pycharm/

