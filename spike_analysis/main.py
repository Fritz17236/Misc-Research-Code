"""
spike_analysis Entry point for experimental data access & analysis.

"""


import scripts
import matplotlib.pyplot as plt
import utils
import os

if __name__ == '__main__':
    session_data_dict = scripts.load_all_session_data(verbose=False)

    for sess_name, sess_data in session_data_dict.items():

        idx_neuron_1 = 10
        idx_neuron_2 = -10
        rng = 10
        bin_width = .001
        lags, cch, num_trials = scripts.trial_averaged_cch(
            session=sess_data, idx_neuron_1=idx_neuron_1,
            idx_neuron_2=idx_neuron_2, rng=rng, bin_width=bin_width
        )

        plt.bar(lags, cch, width=-bin_width, align='edge')
        plt.show()
        exit(0)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

