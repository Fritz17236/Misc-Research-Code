import os
import h5py
import numpy as np
from constants import ENUM_LICK_LEFT, ENUM_LICK_RIGHT, ENUM_NO_LICK, THRESHOLD_FIRING_RATE_MIN
from utils import get_session_firing_rates, load_mat, filter_by_firing_rate
from sklearn.decomposition import PCA


def access_hd5(func):
    def _wrapper_access_hd5_check(*args, **kwargs):
        try:

            return func(*args, **kwargs)
        except KeyError:
            print("No data found. Ensure data loaded into session.")
            raise
    return _wrapper_access_hd5_check

class PersistentObject:
    """
    The persistent object class is used to store objects in structured directories
    in nonvolatile memory.

    """
    EXT_FILE_RECORD = ".record"

    def __init__(self, name, root='.'):
        """
        Construct a PersistentObject Instance.

        The constructor is used to both create new and load existing PersistentObject instances.
        To load an instance, use the name and root corresponding to an existing database. If name
        and root don't refer to an existing instance, a new instance is created.

        :param name: Name of instance, becomes the name of the directory used to store object data.
        :type name: string
        :param root: Root directory to place object instance in
        :type root: string
        """
        self._name = name
        self._root = root
        self._file_list = []
        self._dir = os.path.join(self._root, self._name)
        self._record_path = os.path.join(self._dir, PersistentObject.EXT_FILE_RECORD)
        self._loaded = False
        if not os.path.exists(self._root):
            os.mkdir(self._root)

        if os.path.exists(self._dir):
            if os.path.exists(self._record_path):
                try:
                    self._load()
                except FileNotFoundError:
                    print("No record file found. Is the directory empty?")
                    raise
            else:
                self.delete()
                print("Found Instance {0} with name {1} in root {2}"
                      ", but record was corrupted. Creating new instance...".format(type(self), self._name, self._root))
        else:
            print("No {0} instance with name {1} found in root {2}."
                  " Creating New Instance".format(type(self), self._name, self._root))
            self._create()

        self._loaded = True

    def _load(self):
        """
        Load an existing instance.

        """
        temp_handle = open(self._record_path, 'r')
        self._file_list = [f.rstrip() for f in temp_handle.readlines()]
        temp_handle.close()
        self._record_handle = open(self._record_path, 'a')

    def _create(self):
        """
        Create a new PersistentObject Instance.

        """
        if os.path.exists(self._dir):
            raise FileExistsError("Dir already exists at path: {0}".format(self._dir))
        else:
            os.mkdir(self._dir)
            self._record_handle = open(self._record_path, 'w')

    def contains(self, file_name):
        """
        Check if this instance contains a file

        :param file_name: Name of the file within its parent directory
        :type file_name: string
        :return: True if file contained within this instance, False otherwise.
        """
        return file_name in self._file_list

    def _add_to_file_record(self, file_name):
        """
        Add a file to this instance record.

        If the file is already contained within the directory, this function does nothing.
        Inheriting classes should overload this method to save additional data,
        then call super()._add_to_file_record(file_name) in the overloaded method.
        :param file_name: Name of the file within its parent directory
        :type file_name: string
        """
        if self.contains(file_name):
            print("File {0} already found in record".format(file_name))
            return
        else:
            self._file_list.append(file_name)
            print(file_name, file=self._record_handle)

    def _get_file_record(self):
        """
        Get the record of files stored in this instance.


        :return: file_list
        :rtype: list<string>
        """
        return self._file_list

    def delete(self):
        """
        Delete this instance and its contents.

        Remove the directory containing this instance and its data.
        """
        # TODO: implement delete function
        print("Delete not implemented. Delete {0} manually.".format(self._name))

    def __del__(self):
        if self._loaded:
            self._record_handle.close()


# TODO: handle files individually:
#    for saving: get all fields from a given file, store them in a single data file
#   for loading: just get a handle to the file, store that files fields in  data file
#   when calling accessor, just load all files and concatenate the desired field to return
#   cache in the same instance.


class Session(PersistentObject):
    """
    Sessions store and manage experimental data for persistent access.

    """

    def __init__(self, name, root='.'):
        """
        Construct a session object.

        Session instances are initially empty and data must be added via add_data method.
        :param name: Name of the session object
        :param root: Directory containing session's saved data
        """
        super().__init__(name, root)
        self._data_path = os.path.join(self._dir, "data")

    def add_data(self, file_name, file_directory, **kwargs):
        """
        Add raw data in the form of .mat files to the session object.

        Load .mat files, compute the firing rates and populate session information.
        If session instance contains no data, the first loaded session specifies fields that must
        be the same between sessions (num trials, ts). Additional session data increases the number
        of neurons and the neuron info.

        :param file_name: Name of file to add
        :param file_directory: Name of directory containing file to add
        """
        # TODO: Add support for kwargs to configure how firing rate computed (bin width, firing rate threshold, etc)
        # TODO: Add spike_times field (need ragged array storage)
        # TODO: Check to ensure files are compatible / from same session
        # TODO: Decompose into data-field-specific functions for extraction to improve clarity
        with h5py.File(self._data_path, "a") as data_handle:
            if kwargs:
                raise NotImplementedError("Keyword arguments not yet implemented for add_data")

            #  Raw File I/O
            if self.contains(file_name):
                print("Session already contains data from {0} - skipping.".format(file_name))
                return

            file_path = os.path.join(file_directory, file_name)
            try:
                data = load_mat(file_path)
            except OSError:
                print("Could not load file {0}".format(file_path))
                raise

            # Compute FIring Rates & Filter Neurons
            ts, firing_rates = get_session_firing_rates(data)
            firing_rates, keep_neurons = filter_by_firing_rate(firing_rates, threshold=THRESHOLD_FIRING_RATE_MIN)

            # Populate data for accessor methods
            if "ts" not in data_handle.keys():
                data_handle["ts"] = ts

            if "firing_rates" in data_handle.keys():
                to_add = np.concatenate(
                    (data_handle["firing_rates"], firing_rates), axis=2)
                num_neurons = to_add.shape[2]
                fr_data = data_handle["firing_rates"]
                fr_data.resize((firing_rates.shape[0],
                                firing_rates.shape[1],
                                num_neurons
                                ))
                fr_data[...] = to_add
            else:
                data_handle.create_dataset(name='firing_rates',
                                                 shape=firing_rates.shape,
                                                 maxshape=(firing_rates.shape[0],
                                                           firing_rates.shape[1],
                                                           None), data=firing_rates)

            # extract ragged array, restructure as 2d array of arrays (numpy dtype = object)
            spike_times = np.stack(data["neuron_single_units"][keep_neurons])
            num_neurons_this_sess = len(spike_times)
            num_trials = len(spike_times[0])
            for idx_neuron in range(num_neurons_this_sess):
                for idx_trial in range(num_trials):
                    try:
                        spike_times[idx_neuron][idx_trial].shape
                    except AttributeError:
                        spike_times[idx_neuron][idx_trial] = np.array(spike_times[idx_neuron][idx_trial], ndmin=1)

            if "spike_times" in data_handle.keys():
                to_add = np.concatenate(
                    (data_handle["spike_times"], spike_times), axis=0)
                num_neurons = to_add.shape[0]
                spike_data = data_handle["spike_times"]
                spike_data.resize((num_neurons,
                                num_trials,
                                ))
                spike_data[...] = to_add
            else:
                # spike times are ragged arrays with shape [num_neurons][num_trials][var_num_spikes]
                # requires special treatment from h5py
                dtype = h5py.vlen_dtype(np.float64)
                data_handle.create_dataset(name="spike_times",
                                                 shape=(num_neurons_this_sess, num_trials,),
                                                 maxshape=(None, num_trials,),
                                                 dtype=dtype,
                                                 data=spike_times
                                                 )

            if "lick_directions" not in data_handle.keys():
                lds = np.ones((self.get_num_trials(),)) * ENUM_NO_LICK
                lds[data['task_trial_type'] == 'l'] = ENUM_LICK_LEFT
                lds[data['task_trial_type'] == 'r'] = ENUM_LICK_RIGHT
                data_handle["lick_directions"] = lds

            record = np.core.records.fromarrays(data["neuron_unit_info"][keep_neurons].transpose(),
                                                names='unit_id, unit_quality, unit_x_in_um,'
                                                      ' depth_in_um, associated_electrode, shank, cell_type,'
                                                      ' recording_location',
                                                formats='i4, S10, f4, f4, i4, i4, S10, S32')

            if "neuron_unit_info" not in data_handle.keys():
                data_handle.create_dataset(name="neuron_unit_info", maxshape=(None,), data=record)
            else:
                to_add = np.hstack((data_handle["neuron_unit_info"], record))
                data_handle["neuron_unit_info"].resize((len(to_add),))
                data_handle["neuron_unit_info"][...] = to_add

            self._add_to_file_record(file_name)

    @access_hd5
    def get_spike_times(self):
        with h5py.File(self._data_path, "r") as data_handle:
            data =  data_handle["spike_times"]
        return data


    @access_hd5
    def get_num_neurons(self):
        with h5py.File(self._data_path, "r") as data_handle:
            data = data_handle["firing_rates"].shape[2]
        return data

    @access_hd5
    def get_num_trials(self):
        with h5py.File(self._data_path, "r") as data_handle:
            data = data_handle["firing_rates"].shape[1]
        return data

    @access_hd5
    def get_ts(self):
        with h5py.File(self._data_path, "r") as data_handle:
            data = np.array(data_handle["ts"])
        return data


    @access_hd5
    def get_lick_directions(self):
        with h5py.File(self._data_path, "r") as data_handle:
            data = np.array(data_handle["lick_directions"]).copy().astype(object)
            data[data == ENUM_LICK_LEFT] = 'l'
            data[data == ENUM_LICK_RIGHT] = 'r'
        return data

    @access_hd5
    def get_firing_rates(self, region='all'):
        with h5py.File(self._data_path, "r") as data_handle:
            if region == 'all':
                return np.array(data_handle["firing_rates"])
            elif region in self.get_session_brain_regions():
                neurons_in_this_region = self.get_neuron_brain_regions() == region
                return np.array(data_handle["firing_rates"][:, :, neurons_in_this_region])
            else:
                raise KeyError("Region {0} not contained"
                               " within this session: {1}".format(region, self.get_session_brain_regions()))

    @access_hd5
    def get_neuron_info(self):
        with h5py.File(self._data_path, "r") as data_handle:
            return np.array(data_handle["neuron_unit_info"])

    @access_hd5
    def get_neuron_brain_regions(self):
        """
        Get the location of each neuron contained in the session.

        """
        return np.array(self.get_neuron_info()['recording_location'], dtype=np.str)

    @access_hd5
    def get_session_brain_regions(self):
        """
        Get a list of brain regions contained in this session.

        """
        return np.unique(self.get_neuron_brain_regions())

    @access_hd5
    def get_pca_by_region(self, region='left ALM'):
        """
        Get the results from principal components analysis

        :param region:
        :return: (components, projections) components: the principal components themselves,
         projections: the firing rates projected onto components.
        """
        with h5py.File(self._data_path, "r") as data_handle:
            if region in self.get_session_brain_regions():
                return data_handle["pca_components" + "/" + region], data_handle[
                    "pca_projections" + "/" + region]
            else:
                raise KeyError("Region {0} not contained"
                               " within this session: {1}".format(region, self.get_session_brain_regions()))



    def compute_firing_rate_pca_by_brain_region(self, num_pcs=2):
        """
        Perform Principal Component Analysis on stored Firing Rates, Splitting by brain region

        PCA is performed across trials
        :param num_pcs: number of principal components to compute.
        :return:
        """
        # TODO: add overwrite toggle if clients want to recompute with different number of pcs on same session instance
        data_handle = h5py.File(self._data_path, 'a')
        with h5py.File(self._data_path, "r") as data_handle:

            regions = self.get_session_brain_regions()
            print("Performing Principal Component Analysis...", end='')

            # check if pcs already exist, if so delete and recompute
            if "pca_projections" in data_handle.keys() and "pca_components" in data_handle.keys():
                print("PCA Already Performed on Session {0} - skipping.".format(self._name))
                return
            for region in regions:
                frs = self.get_firing_rates(region=region)
                (num_bins, num_trials, num_neurons) = frs.shape
                frs_concat = np.swapaxes(frs, 0, 2).reshape((num_neurons, num_bins * num_trials))
                pca = PCA(n_components=num_pcs, svd_solver ="full")
                pca.fit(frs_concat.T)
                fr_pcas = np.zeros((num_bins, num_trials, num_pcs))

                for j in range(num_pcs):
                    component = pca.components_[j, :]
                    fr_pcas[:, :, j] = np.tensordot(frs, component, axes=1)

                data_handle["pca_projections" + "/" + region] = fr_pcas
                data_handle["pca_components" + "/" + region] = pca.components_

                print("done.")


# OLD CODE
# def _save(self):
#     print("Saving Instance {0}...".format(self._name), end='')
#     data_to_save = { field : getattr(self, field) for (field, _) in Session.FIELDS}
#     print('done.')
#     for (field,_) in Session.FIELDS:
#         data = getattr(self, field)
#         save_path = os.path.join(self._dir, self._name + "_" + field)
#         np.save(save_path, data)
# os.rename(save_path + ".npy", save_path + Session.EXT_SESSION_DATA)
# def _save_file_data(self, file_name, file_data):
#     """
#     Save file data, separating fields into different files.
#
#     :param file_name: name of the file being saved, used to name, prepends data being saved
#     :type file_name: string
#     :param file_data: dictionary [Fields]-->file data, field nae suffixes data being saved. Must contain
#     :type file_data: dict
#     an entry for each FIELD specified in Session.FIELDS
#     """
#     # first check that file_data specifies each field
#     for (f, _) in Session.FIELDS:
#         assert(f in file_data.keys())
#
#     for (f, _) in Session.FIELDS:
#         save_name = file_name + "/" + f
#         dset = data_handle.create_dataset(name=save_name, data=file_data[f])
#
# def _load_file_data(self, file_name):
#     """
#     Load all field data from a given file.
#
#     :param file_name: name of file to load
#     :type file_name: string
#     :return: file_data, dictionary with keys=field and value=data for given file
#     :rtype: dict
#     """
#     file_data = {}
#     for (f, _) in Session.FIELDS:
#        load_name = file_name + "_" + f + Session.EXT_SESSION_DATA
#        load_path = os.path.join(self._dir, load_name)
#        field_data = np.load(load_path)
#        file_data[f] = field_data
#     return file_data
#
# def _load_field_by_file(self, field, file_name):
#     """
#     Load given field data for a given file.
#
#     :param field: The field to load. must be listed in Session.FIELD
#     :type field: string
#     :param file_name: the name of the file to load. must be contained within the Session filed record
#     :type file_name: string
#     :return: data
#     :rtype: variable = Session.FIELDS[field]
#     """
#     if field not in [ f[0] for f in Session.FIELDS]:
#         raise KeyError("Requested field {0} not in Session.FIELDS ".format(field))
#     if not self.contains(file_name):
#         raise FileNotFoundError("File {0} not listing within"
#                                 " file record of {1}".format(file_name, self._name))
#     try:
#         file_path = os.path.join(self._dir, file_name + "_" + field + Session.EXT_SESSION_DATA)
#         return np.load(file_path)
#     except OSError:
#         print("Error loading file {0}".format(file_path))
#         raise
#
# def _load_field(self, field):
#     """
#     Load field data from all fields in a given record.
#
#     :param field: Field name of data to load
#     :type field: string
#     :return: field data
#     :rtype: variable = Session.FIELDS[field]
#     """
#     # ('firing_rates', np.ndarray),
#     # ('num_neurons', np.int32),
#     # ('num_trials', np.int32),
#     # ('neurons_info', np.ndarray),
#     # ('ts', np.ndarray),
#     # ('spike_times', np.ndarray),
#     # ('lick_dirs', np.ndarray)
#     if field == "firing_rates":
#         pass
#     # if field needs to be stacked from multiple files
#         # get a list of all files of that field in this sessions
#     # else just load one file


#
# # region ## Load PCA Data
# pcas = []
# for frs in os.listdir(DIR_PCA_SAVE):
#     if frs.endswith("_pca_two.npy"):
#         pcas.append(np.load(DIR_PCA_SAVE + "\\" + frs))
# pcas = pcas[:4]
#
# ts = np.load(DIR_FIRING_RATES + "/" + "map-export_SC022_20190228_140832_s2_p1.mat_binCenters.npy")
# # endregion
#
# # region ## Plot First Two PCAs ##
#
# pc1 = pcas[0]
# num_ts = pc1.shape[0]
# num_trials = pc1.shape[1]
#
# idx_trial = 10
# titles = ["PC One Left ALM", "PC One Right ALM", "PC One Left Striatum", "PC One Right Striatum"]
#
# idx_pc = 0
# for j, title in enumerate(titles):
#     pc1 = pcas[j]
#     pc1s = np.zeros((num_ts, num_trials))
#     for i in range(num_trials):
#         pc1s[:, i] = pc1[:, i, idx_pc]
#     error = np.std(pc1s, axis=1) / np.sqrt(num_trials)
#     y = np.mean(pc1s, axis=1)
#     plt.figure(title)
#     plt.plot(ts, y, label='Mean +/- S.E.M', c='g')
#     plt.fill_between(ts, y - error, y + error, alpha=.2, color='g')
#     plt.plot(ts, pc1s[:, idx_trial], c='r', label='Trial {0}'.format(idx_trial))
#     plt.title(title)
#     plt.ylabel("Firing Rate")
#     plt.xlabel("Time (s)")
#     plt.legend()
#     plt.savefig(title + ".png", dpi=128, bbox_inches='tight')
#     plt.show()
#
# titles = ["PC Two Left ALM", "PC Two Right ALM", "PC Two Left Striatum", "PC Two Right Striatum"]
#
# idx_pc = 1
# for j, title in enumerate(titles):
#     pc1 = pcas[j]
#     pc1s = np.zeros((num_ts, num_trials))
#     for i in range(num_trials):
#         pc1s[:, i] = pc1[:, i, idx_pc]
#     error = np.std(pc1s, axis=1) / np.sqrt(num_trials)
#     y = np.mean(pc1s, axis=1)
#     plt.figure(title)
#     plt.plot(ts, y, label='Mean +/- S.E.M', c='g')
#     plt.fill_between(ts, y - error, y + error, alpha=.2, color='g')
#     plt.plot(ts, pc1s[:, idx_trial], c='r', label='Trial {0}'.format(idx_trial))
#     plt.title(title)
#     plt.ylabel("Firing Rate")
#     plt.xlabel("Time (s)")
#     plt.legend()
#     plt.savefig(title + ".png", dpi=128, bbox_inches='tight')
#     plt.show()
# # endregion
#
# # region ## Trial-Average PCA Cross-Correlation
# # fs = 1 / (ts[1] - ts[0])
# # lags = ts_to_acorr_lags(ts)
# #
# # data_sess = list_data_sess[0]
# # left_lick_trials = get_trials_by_lick_direction(data_sess, 'l')
# # right_lick_trials = get_trials_by_lick_direction(data_sess, 'r')
# #
# # trials = right_lick_trials
# #
# # num_trials = len(trials)
# #
# #
# # crosscorrs_alm_left_pca_one_alm_left_pca_two = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_alm_right_pca_one = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_alm_right_pca_two = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_stri_left_one = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_stri_left_two = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_stri_right_one = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_stri_right_two = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_shifted = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_arma = np.zeros((len(lags), num_trials))
# #
# #
# #
# # crosscorrs_alm_left_pca_one_alm_left_pca_two_whitened = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_alm_right_pca_one_whitened = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_alm_right_pca_two_whitened = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_stri_left_one_whitened = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_stri_left_two_whitened = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_stri_right_one_whitened = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_stri_right_two_whitened = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_pca_one_shifted_whitened = np.zeros((len(lags), num_trials))
# # crosscorrs_alm_left_arma_whitened = np.zeros((len(lags), num_trials))
# #
# # for i in range(num_trials):
# #     idx_trial = trials[i]
# #     #pcas look like (num bins, num trials, num pca components)
# #     pca_alm_left_one = pcas[0][:, idx_trial, 0]
# #     pca_alm_left_two = pcas[0][:, idx_trial, 1]
# #     pca_alm_right_one = pcas[1][:, idx_trial, 0]
# #     pca_alm_right_two = pcas[1][:, idx_trial, 1]
# #     pca_stri_left_one = pcas[2][:, idx_trial, 0]
# #     pca_stri_left_two = pcas[2][:, idx_trial, 1]
# #     pca_stri_right_one = pcas[3][:, idx_trial, 0]
# #     pca_stri_right_two = pcas[3][:, idx_trial, 1]
# #
# #     space = 10
# #     pca_alm_left_arma = 3 * np.roll(pca_alm_left_one, -1 * space) + 2 * np.roll(pca_alm_left_one, -2 * space) + 1 * np.roll(pca_alm_left_one, -3 * space)
# #
# #     pca_alm_left_one_shifted = np.roll(pca_alm_left_one, 101) #+ np.random.normal(scale=0, size=pca_alm_left_one_shifted.shape)
# #
# #     freqs, Filt = get_whitening_filter(pca_alm_left_one, fs, b=np.inf, mode='highpass')
# #
# #
# #     pca_alm_left_one_whitened = apply_filter(pca_alm_left_one, Filt)
# #     pca_alm_left_two_whitened = apply_filter(pca_alm_left_two, Filt)
# #     pca_alm_right_one_whitened = apply_filter(pca_alm_right_one, Filt)
# #     pca_alm_right_two_whitened = apply_filter(pca_alm_right_two, Filt)
# #     pca_stri_left_one_whitened = apply_filter(pca_stri_left_one, Filt)
# #     pca_stri_left_two_whitened = apply_filter(pca_stri_left_two, Filt)
# #     pca_stri_right_one_whitened = apply_filter(pca_stri_right_one, Filt)
# #     pca_stri_right_two_whitened = apply_filter(pca_stri_right_two, Filt)
# #     pca_alm_left_one_shifted_whitened = apply_filter(pca_alm_left_one_shifted, Filt)
# #     pca_alm_left_arma_whitened = apply_filter(pca_alm_left_arma, Filt)
# #
# #     crosscorrs_alm_left_pca_one_alm_left_pca_two[:,i] = cross_correlation(pca_alm_left_one, pca_alm_left_two)
# #     crosscorrs_alm_left_pca_one_alm_right_pca_one[:,i] = cross_correlation(pca_alm_left_one, pca_alm_right_one)
# #     crosscorrs_alm_left_pca_one_alm_right_pca_two[:,i] =  cross_correlation(pca_alm_left_one, pca_alm_right_two)
# #     crosscorrs_alm_left_pca_one_stri_left_one[:,i] = cross_correlation(pca_alm_left_one, pca_stri_left_one)
# #     crosscorrs_alm_left_pca_one_stri_left_two[:,i] = cross_correlation(pca_alm_left_one, pca_stri_left_two)
# #     crosscorrs_alm_left_pca_one_stri_right_one[:,i] = cross_correlation(pca_alm_left_one, pca_stri_right_one)
# #     crosscorrs_alm_left_pca_one_stri_right_two[:,i] = cross_correlation(pca_alm_left_one,  pca_stri_right_two)
# #     crosscorrs_alm_left_pca_one_shifted[:,i] = cross_correlation(pca_alm_left_one,  pca_alm_left_one_shifted)
# #     crosscorrs_alm_left_arma[:,i] = cross_correlation(pca_alm_left_one,  pca_alm_left_arma)
# #
# #
# #     crosscorrs_alm_left_pca_one_alm_left_pca_two_whitened[:,i] = cross_correlation(pca_alm_left_one_whitened, pca_alm_left_two_whitened)
# #     crosscorrs_alm_left_pca_one_alm_right_pca_one_whitened[:,i] = cross_correlation(pca_alm_left_one_whitened, pca_alm_right_one_whitened)
# #     crosscorrs_alm_left_pca_one_alm_right_pca_two_whitened[:,i] =  cross_correlation(pca_alm_left_one_whitened, pca_alm_right_two_whitened)
# #     crosscorrs_alm_left_pca_one_stri_left_one_whitened[:,i] = cross_correlation(pca_alm_left_one_whitened, pca_stri_left_one_whitened)
# #     crosscorrs_alm_left_pca_one_stri_left_two_whitened[:,i] = cross_correlation(pca_alm_left_one_whitened, pca_stri_left_two_whitened)
# #     crosscorrs_alm_left_pca_one_stri_right_one_whitened[:,i] = cross_correlation(pca_alm_left_one_whitened, pca_stri_right_one_whitened)
# #     crosscorrs_alm_left_pca_one_stri_right_two_whitened[:,i] = cross_correlation(pca_alm_left_one_whitened, pca_stri_right_two_whitened)
# #     crosscorrs_alm_left_pca_one_shifted_whitened[:,i] = cross_correlation(pca_alm_left_one_whitened,  pca_alm_left_one_shifted_whitened)
# #     crosscorrs_alm_left_arma_whitened[:,i] = cross_correlation(pca_alm_left_one_whitened,  pca_alm_left_arma_whitened)
# #
# # ylims = [-1, 1]
# #
# #
# # plt.figure("Cross-Correlation ALM Left PC One, Itself Shifted {0} sec, N = {1} Trials".format(np.round(100 / fs), num_trials))
# # plt.title("Cross-Correlation ALM Left PC One, Itself Shifted {0} sec, N = {1} Trials".format(np.round(100 / fs), num_trials))
# # error = np.std(crosscorrs_alm_left_pca_one_shifted, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_shifted, axis=1)
# # plt.plot(lags, y, label='Unwhitened +/- S.E.M', c='r')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='r')
# #
# # error = np.std(crosscorrs_alm_left_pca_one_shifted_whitened, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_shifted_whitened, axis=1)
# # plt.plot(lags, y, label='Whitened +/- S.E.M', c='g')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='g')
# # plt.ylabel("Cross-correlation Coefficient")
# # plt.xlabel(r"Time Lag $\tau$ (s)")
# # plt.legend()
# # plt.ylim(ylims)
# # plt.savefig("./figs/Cross-Correlation ALM Left PC One, Itself Shifted.png", dpi=128, bbox_inches='tight')
# # plt.show()
# #
# #
# # spaces = [np.round(j * space / fs, 2)  for j in [1, 2, 3]]
# # plt.figure("Cross-Correlation ALM Left PC One, AR Model, $Y(t) = 3 x(t - {0}) + 2 x(t - {1}) + 1 x(t - {2})$".format(*spaces))
# # plt.title("Cross-Correlation ALM Left PC One, AR Model, $Y(t) = 3 x(t - {0}) + 2 x(t - {1}) + 1 x(t - {2})$".format(*spaces))
# # error = np.std(crosscorrs_alm_left_arma, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_arma, axis=1)
# # plt.plot(lags, y, label='Unwhitened +/- S.E.M', c='r')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='r')
# # error = np.std(crosscorrs_alm_left_arma_whitened, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_arma_whitened, axis=1)
# # plt.plot(lags, y, label='Whitened +/- S.E.M', c='g')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='g')
# # plt.ylabel("Cross-correlation Coefficient")
# # plt.xlabel(r"Time Lag $\tau$ (s)")
# # plt.legend()
# # plt.ylim(ylims)
# # plt.savefig("./figs/Cross-Correlation ALM Left PC One, AR Model.png", dpi=128, bbox_inches='tight')
# # plt.show()
# #
# #
# # plt.figure("Cross-Correlation ALM Left PC One, ALM Right PC One, N = {0} Trials".format(num_trials))
# # plt.title("Cross-Correlation ALM Left PC One, ALM Right PC One, N = {0} Trials".format(num_trials))
# # error = np.std(crosscorrs_alm_left_pca_one_alm_right_pca_one, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_alm_right_pca_one, axis=1)
# # plt.plot(lags, y, label='Unwhitened +/- S.E.M', c='r')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='r')
# #
# # error = np.std(crosscorrs_alm_left_pca_one_alm_right_pca_one_whitened, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_alm_right_pca_one_whitened, axis=1)
# # plt.plot(lags, y, label='Whitened +/- S.E.M', c='g')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='g')
# # plt.ylabel("Cross-correlation Coefficient")
# # plt.xlabel(r"Time Lag $\tau$ (s)")
# # plt.legend()
# # plt.ylim(ylims)
# # plt.savefig("Cross-Correlation ALM Left PC One, ALM Right PC One, N = {0} Trials".format(num_trials), dpi=128, bbox_inches='tight')
# # plt.show()
# #
# # plt.figure("Cross-Correlation ALM Left PC One, ALM Right PC Two, N = {0} Trials".format(num_trials))
# # plt.title("Cross-Correlation ALM Left PC One, ALM Right PC Two, N = {0} Trials".format(num_trials))
# # error = np.std(crosscorrs_alm_left_pca_one_alm_right_pca_two, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_alm_right_pca_two, axis=1)
# # plt.plot(lags, y, label='Unwhitened +/- S.E.M', c='r')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='r')
# #
# # error = np.std(crosscorrs_alm_left_pca_one_alm_right_pca_two_whitened, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_alm_right_pca_two_whitened, axis=1)
# # plt.plot(lags, y, label='Whitened +/- S.E.M', c='g')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='g')
# # plt.ylabel("Cross-correlation Coefficient")
# # plt.xlabel(r"Time Lag $\tau$ (s)")
# # plt.legend()
# # plt.ylim(ylims)
# # plt.savefig("Cross-Correlation ALM Left PC One, ALM Right PC Two, N = {0} Trials".format(num_trials), dpi=128, bbox_inches='tight')
# # plt.show()
# #
# #
# #
# #
# # plt.figure("Cross-Correlation ALM Left PC One, ALM Left PC Two")
# # plt.title("Cross-Correlation ALM Left PC One, ALM Left PC Two,  N = {0} Trials".format(num_trials))
# # error = np.std(crosscorrs_alm_left_pca_one_alm_left_pca_two, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_alm_left_pca_two, axis=1)
# # plt.plot(lags, y, label='Unwhitened +/- S.E.M', c='r')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='r')
# #
# # error = np.std(crosscorrs_alm_left_pca_one_alm_left_pca_two_whitened, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_alm_left_pca_two_whitened, axis=1)
# # plt.plot(lags, y, label='Whitened +/- S.E.M', c='g')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='g')
# # plt.ylabel("Cross-correlation Coefficient")
# # plt.xlabel(r"Time Lag $\tau$ (s)")
# # plt.legend()
# # plt.ylim(ylims)
# # plt.savefig("./figs/Cross-Correlation ALM Left PC One, ALM Left PC Two,  N = {0} Trials.png".format(num_trials), dpi=128, bbox_inches='tight')
# #
# # plt.show()
# #
# #
# # plt.figure("Cross-Correlation ALM Left PC One, Striatum Left PC One")
# # plt.title("Cross-Correlation ALM Left PC One, Striatum Left PC One ,  N = {0} Trials.png".format(num_trials))
# # error = np.std(crosscorrs_alm_left_pca_one_stri_left_one, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_stri_left_one, axis=1)
# # plt.plot(lags, y, label='Unwhitened +/- S.E.M', c='r')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='r')
# #
# # error = np.std(crosscorrs_alm_left_pca_one_stri_left_one_whitened, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_stri_left_one_whitened, axis=1)
# # plt.plot(lags, y, label='Whitened +/- S.E.M', c='g')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='g')
# # plt.ylabel("Cross-correlation Coefficient")
# # plt.xlabel(r"Time Lag $\tau$ (s)")
# # plt.legend()
# # plt.ylim(ylims)
# # plt.savefig("./figs/Cross-Correlation ALM Left PC One, Striatum Left PC One ,  N = {0} Trials.png".format(num_trials), dpi=128, bbox_inches='tight')
# #
# # plt.show()
# #
# #
# # plt.figure("Cross-Correlation ALM Left PC One, Striatum Left PC Two")
# # plt.title("Cross-Correlation ALM Left PC One, Striatum Left PC Two,  N = {0} Trials.png".format(num_trials))
# # error = np.std(crosscorrs_alm_left_pca_one_stri_left_two, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_stri_left_two, axis=1)
# # plt.plot(lags, y, label='Unwhitened +/- S.E.M', c='r')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='r')
# #
# # error = np.std(crosscorrs_alm_left_pca_one_stri_left_two_whitened, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_stri_left_two_whitened, axis=1)
# # plt.plot(lags, y, label='Whitened +/- S.E.M', c='g')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='g')
# # plt.ylabel("Cross-correlation Coefficient")
# # plt.xlabel(r"Time Lag $\tau$ (s)")
# # plt.legend()
# # plt.ylim(ylims)
# # plt.savefig("./figs/Cross-Correlation ALM Left PC One, Striatum Left PC Two,  N = {0} Trials.png".format(num_trials), dpi=128, bbox_inches='tight')
# #
# # plt.show()
# #
# #
# # plt.figure("Cross-Correlation ALM Left PC One, Striatum Right PC One")
# # plt.title("Cross-Correlation ALM Left PC One, Striatum Right PC One,  N = {0} Trials".format(num_trials))
# # error = np.std(crosscorrs_alm_left_pca_one_stri_right_one, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_stri_right_one, axis=1)
# # plt.plot(lags, y, label='Unwhitened +/- S.E.M', c='r')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='r')
# #
# # error = np.std(crosscorrs_alm_left_pca_one_stri_right_one_whitened, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_stri_right_one_whitened, axis=1)
# # plt.plot(lags, y, label='Whitened +/- S.E.M', c='g')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='g')
# # plt.ylabel("Cross-correlation Coefficient")
# # plt.xlabel(r"Time Lag $\tau$ (s)")
# # plt.legend()
# # plt.ylim(ylims)
# # plt.savefig("./figs/Cross-Correlation ALM Left PC One, Striatum Right PC One,  N = {0} Trials.png".format(num_trials), dpi=128, bbox_inches='tight')
# # plt.show()
# #
# # plt.figure("Cross-Correlation ALM Left PC One, Striatum Right PC Two")
# # plt.title("Cross-Correlation ALM Left PC One, Striatum Right PC Two,  N = {0} Trials".format(num_trials))
# # error = np.std(crosscorrs_alm_left_pca_one_stri_right_two, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_stri_right_two, axis=1)
# # plt.plot(lags, y, label='Unwhitened +/- S.E.M', c='r')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='r')
# #
# # error = np.std(crosscorrs_alm_left_pca_one_stri_right_two_whitened, axis = 1) / np.sqrt(num_trials)
# # y = np.mean(crosscorrs_alm_left_pca_one_stri_right_two_whitened, axis=1)
# # plt.plot(lags, y, label='Whitened +/- S.E.M', c='g')
# # plt.fill_between(lags, y-error, y+error, alpha=.2, color='g')
# # plt.ylabel("Cross-correlation Coefficient")
# # plt.xlabel(r"Time Lag $\tau$ (s)")
# # plt.legend()
# # plt.ylim(ylims)
# # plt.savefig("./figs/Cross-Correlation ALM Left PC One, Striatum Right PC Two.png", dpi=128, bbox_inches='tight')
# # plt.show()
# # endregion
