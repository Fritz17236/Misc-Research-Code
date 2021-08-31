"""
containers.py Class definitions for persistent data storage with speedy access.
"""

import os
import h5py
import numpy as np

import constants
import utils
from constants import ENUM_LICK_LEFT, ENUM_LICK_RIGHT, ENUM_NO_LICK, THRESHOLD_FIRING_RATE_MIN
from utils import get_session_firing_rates, load_mat, filter_by_firing_rate
from sklearn.decomposition import PCA


# Wrapper function used when accessing data
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
    Store data in a structured directory.

    The persistent object class is used to store objects in structured directories
    in nonvolatile memory. It also keeps track of files that have been added to prevent
    needless addition, duplication, or unintended overwriting.
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
        with open(self._record_path, 'r') as record_handle:
            self._file_list = [f.rstrip() for f in record_handle.readlines()]


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
            with open(self._record_path, "a") as f:
                print(file_name, file=f)

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


class Session(PersistentObject):
    """
    Sessions store and manage experimental data for persistent access.

    """

    def __init__(self, name, root='.', verbose=False):
        """
        Construct a session object.

        Session instances are initially empty and data must be added via add_data method.
        :param name: Name of the session object
        :param root: Directory containing session's saved data
        """
        super().__init__(name, root)
        self.verbose = verbose
        self._data_path = os.path.join(self._dir, "data")

    def add_data(self, file_name, file_directory, **kwargs):
        """
        Add raw data in the form of .mat files to the session object.

        Load .mat files, compute the firing rates and populate session information.
        If session instance contains no data, the first loaded session specifies fields that must
        be the same between sessions (num trials, ts). Additional session data increases the number
        of neurons and the neuron info record length.

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
                if self.verbose:
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

            if "task_stimulation" not in data_handle.keys():
                data_handle.create_dataset(name="task_stimulation", data=data['task_stimulation'])

            if "task_cue_time" not in data_handle.keys():
                data_handle.create_dataset(name="task_cue_time", data=data['task_cue_time'])

            if "task_delay_time" not in data_handle.keys():
                data_handle.create_dataset(name="task_delay_time", data=data['task_delay_time'])

            if "task_sample_time" not in data_handle.keys():
                data_handle.create_dataset(name="task_sample_time", data=data['task_sample_time'])

            # TODO: add task stimulation, task cue time, task delay time, filter by epoch
        self._add_to_file_record(file_name)

    # Accessor methods (necessary to use wrapper?)
    @access_hd5
    def get_spike_times(self):
        """
        Get the spike times of the neurons as a 2d sparse array [num_neurons][num_trials].

        :return:
        """
        with h5py.File(self._data_path, "r") as data_handle:
            data = np.asarray(data_handle["spike_times"])
        return data

    @access_hd5
    def get_task_stimulation(self):
        """
        Get the stimulation parameters for a given trial

        :return: stims (2d numpy array of shape [num_trials], 4)
        For each trial, the 4 columns are [
        laser_power,
         stim_type('1', '2', or '6', with 1,2,6 being left/right/both ALM perturbation),
          laser_on_time,
           laser_off_time]
            (with reference to trial start time - need to subtract go cue time from it).
        """
        with h5py.File(self._data_path, "r") as data_handle:
            data = np.asarray(data_handle["task_stimulation"])
        return data

    @access_hd5
    def get_task_cue_times(self):
        """
        Get the stimulation parameters for a given trial

        :return: stims (2d numpy array of shape [num_trials], 2)
        First row is go cue time, second row is duration of response period
        """
        with h5py.File(self._data_path, "r") as data_handle:
            data = np.asarray(data_handle["task_cue_time"])
        return data

    @access_hd5
    def get_task_sample_times(self):
        """
        Get the stimulation parameters for a given trial

        :return: stims (2d numpy array of shape [num_trials], 2)
        First row is go sample time, second row is duration of sample period
        """
        with h5py.File(self._data_path, "r") as data_handle:
            data = np.asarray(data_handle["task_sample_time"])
        return data

    @access_hd5
    def get_task_delay_times(self):
        """
        Get the stimulation parameters for a given trial

        :return: stims (2d numpy array of shape [num_trials], 2)
        First row is go delay time, second row is duration of delay period
        """
        with h5py.File(self._data_path, "r") as data_handle:
            data = np.asarray(data_handle["task_delay_time"])
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
    def get_task_lick_directions(self):
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
                return np.array(data_handle["firing_rates"])[:, :, neurons_in_this_region]
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
                return np.asarray(data_handle["pca_components" + "/" + region]), np.asarray(data_handle[
                    "pca_projections" + "/" + region])
            else:
                raise KeyError("Region {0} not contained"
                               " within this session: {1}".format(region, self.get_session_brain_regions()))

    def compute_pca_by_brain_region(self, num_pcs=2, overwrite=False, mode='fr', dt=constants.BIN_WIDTH_DEFAULT,
                                    t_start=constants.TIME_BEGIN_DEFAULT, t_stop=constants.TIME_END_DEFAULT):
        """
        Perform Principal Component Analysis on stored Firing Rates, Splitting by brain region

        PCA is performed across trials
        :param num_pcs: number of principal components to compute.
        :return:
        """

        regions = self.get_session_brain_regions()
        if self.verbose:
            print("Performing Principal Component Analysis...", end='')

        with h5py.File(self._data_path, "a") as data_handle:
            if "pca_projections" in data_handle.keys() and "pca_components" in data_handle.keys():
                if overwrite:
                    del data_handle["pca_projections"]
                    del data_handle['pca_components']
                else:
                    if self.verbose:
                        print("PCA Already Performed on Session {0} - skipping.".format(self._name))
                    return

        ts = self.get_ts()



        for region in regions:
            if mode == 'fr':
                frs = self.get_firing_rates(region=region).copy()
            elif mode == 'isi':
                frs = utils.compute_trial_isis(self, dt, t_start, t_stop)

            (num_bins, num_trials, num_neurons) = frs.shape
            frs_concat = np.swapaxes(frs, 0, 2).reshape((num_neurons, num_bins * num_trials))
            pca = PCA(n_components=num_pcs, svd_solver="full")
            try:
                pca.fit(frs_concat.T)
            except FloatingPointError:
                print("True divide by zero error encountered in pca for session {0}".format(self._name))
            fr_pcas = np.zeros((num_bins, num_trials, num_pcs))

            for j in range(num_pcs):
                component = pca.components_[j, :]
                fr_pcas[:, :, j] = np.tensordot(frs, component, axes=1)

            with h5py.File(self._data_path, "a") as data_handle:
                data_handle["pca_projections" + "/" + region] = fr_pcas
                data_handle["pca_components" + "/" + region] = pca.components_

        print("done.")
