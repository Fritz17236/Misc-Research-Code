# data I/O

from collections import defaultdict
import os
from constants import DIR_RAW, DIR_SAVE
from utils import ts_to_acorr_lags, get_psd, get_whitening_filter, autocorrelation, apply_filter, cross_correlation
import numpy as np
import matplotlib.pyplot as plt
from containers import Session


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
        split_name = fname.split('_')
        mat_header = split_name[0]
        session_name = split_name[1] + "_" + split_name[2] + "_" + split_name[3] + "_" + split_name[4]
        match_pattern = mat_header + "_" + session_name

        for file_name in file_name_list:
            if file_name.startswith(match_pattern) and file_name not in session_name_dict[session_name]:
                session_name_dict[session_name].append(file_name)
    return session_name_dict


def load_all_session_data():
    """
    Load all session data, pre-processing (including PCA w/ 2 components) if necessary

    :return: session_data dictioanry object with keys being session names, and values being Session Instances.
    """

    print("Loading Session Data...")

    session_names = get_session_name_dict(os.listdir(DIR_RAW))
    session_data = {}

    for idx_session, (session_name, files) in enumerate(session_names.items()):
        print("\nprocessing session {0}/{1}...".format(idx_session + 1, len(session_names.keys())))

        #create session instance to store data
        try:
            sess = Session(name=session_name, root=DIR_SAVE)
            for j, file_name in enumerate(files):
                print("\tadding file {0}/{1}".format(j + 1, len(files)))

                # create session object with given name, data saved in DIR_SAVE declared in constants.py
                sess.add_data(file_name=file_name, file_directory=DIR_RAW)

            # now compute principal components with option to specify how many principal components:
            sess.compute_firing_rate_pca_by_brain_region(num_pcs=2)

            # for accessing later in script
            session_data[session_name] = sess
        except Exception as e:
            with open(os.path.join(DIR_SAVE,"log.txt"), "a") as log:
                print("error loading session {0}".format(session_name))
                print("error loading session {0}".format(session_name), file=log)
                print(e)
                print(e, file=log)
                raise


    print("Session Data Loaded.")
    return session_data


# Single-Session Scripts
def plot_pc_autocorrelation_analysis(session, pc_ref, pc_name, save_dir):
    """
    Plot the autocorrelation and power spectra of all prinicpal components in a given region

    :param pc_name:
    :param session: Session instance containing data
    :param ref_pc:  reference principal component used to decide whitening and autocorrelation
    :type session: Session
    """

    fs=1 / (session.get_ts()[1]-session.get_ts()[0])
    lags = ts_to_acorr_lags(session.get_ts())

    for region in session.get_session_brain_regions():

        comps, projs = session.get_pca_by_region(region=region)
        num_bins, num_trials, num_pcs = projs.shape
        freqs, _ = get_psd(pc_ref[:,0 ], fs)
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


            title="Autocorrelation_Region_{0}_PC_{1}_Whitened_wrt_{2}".format(region, idx_pc + 1, pc_name)
            save_path = os.path.join(save_dir, title + ".png")
            plt.figure(title)
            plt.title(title)
            error = np.std(autocorrs, axis=1) / np.sqrt(num_trials)
            y = np.mean(autocorrs, axis=1)
            error_whitened = np.std(autocorrs_whitened, axis=1) / np.sqrt(num_trials)
            y_whitened = np.mean(autocorrs_whitened, axis=1)
            plt.plot(lags, y, label="Unwhitened", c='r')
            plt.plot(lags, y_whitened, label='Whitened',c='g')
            plt.fill_between(lags, y_whitened - error_whitened, y_whitened + error_whitened, alpha=.2, color='r')
            plt.fill_between(lags, y_whitened - error_whitened, y_whitened + error_whitened, alpha=.2, color='g')
            plt.ylabel("Autocorrelation Coefficient")
            plt.xlabel(r"Time Lag $\tau$ (s)")
            plt.ylim([-1, 1])
            plt.legend()
            plt.savefig(save_path, dpi=128, bbox_inches='tight')

            title="Power_Spectral_Density_Region_{0}_PC_{1}_Whitened_wrt_{2}".format(region, idx_pc + 1, pc_name)
            save_path = os.path.join(save_dir, title + ".png")

            plt.figure(title)
            plt.title(title)
            error = np.std(psds, axis=1) / np.sqrt(num_trials)
            y = np.mean(psds, axis=1)
            error_whitened = np.std(psds_whitened, axis=1) / np.sqrt(num_trials)
            y_whitened = np.mean(psds_whitened, axis=1)
            plt.loglog(freqs, y, label="Unwhitened", c='r')
            plt.loglog(freqs, y_whitened, label='Whitened',c='g')
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

# Multi-Session Scripts

