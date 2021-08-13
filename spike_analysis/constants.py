
# Change these to suit your needs
DIR_RAW = r"C:\Users\fritz\Desktop\MAP_data_2021May"  # data directory (mat files)
DIR_SAVE = r"C:\Users\fritz\Desktop\session_data"     # directory to store persistent/computed data
EXT_RAW_DATA = ".mat"  # file extension for raw data added to session


TIME_BEGIN_DEFAULT = -3.0  # trial start time
TIME_END_DEFAULT = 3.5  # trial end time
BIN_WIDTH_DEFAULT = .05
THRESHOLD_FIRING_RATE_MIN = 1  # Lowest firing rate in Hz
ENUM_LICK_LEFT = 1  ##lick directions: -1 (no data), 0 (right), 1 (left)
ENUM_LICK_RIGHT = 0
ENUM_NO_LICK = -1
NEURON_UNIT_INFO_IDX = {
    'unit_id': 0,
    'unit_quality': 1,
    'unit_x_in_um': 2,
    'depth_in_um': 3,
    'associated_electrode': 4,
    'shank': 5,
    'cell_type': 6,  # one of 'Pyr' (Pyramidal Neurons), 'FS' (Fast Spiking), 'not classified', 'all' (all types)
    'recording_location': 7
}
NEURON_UNIT_INFO_DTYPES = [
    ('unit_id', '<i4'),
    ('unit_quality', 'S10'),
    ('unit_x_in_um', '<f4'),
    ('depth_in_um', '<f4'),
    ('associated_electrode', '<i4'),
    ('shank', '<i4'),
    ('cell_type', 'S10'),  # one of 'Pyr' (Pyramidal Neurons), 'FS' (Fast Spiking), 'not classified', 'all' (all types)
    ('recording_location', 'S32')
]
DT_DEFAULT = .001  # default time step (spacing between bins when estimating firing rate)
