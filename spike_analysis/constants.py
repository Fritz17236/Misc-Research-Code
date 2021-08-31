"""
constants.py  Configure default constants for data management and preprocessing
"""

# Change these to suit your needs
DIR_RAW = r"C:\Users\fritz\Desktop\MAP_data_2021May\subset"  # data directory (mat files)
DIR_SAVE = r"C:\Users\fritz\Desktop\session_data\subset"     # directory to store persistent/computed data

# Spike Rate Estimation
TIME_BEGIN_DEFAULT = -3.0  # trial start time
TIME_END_DEFAULT = 3.5  # trial end time
BIN_WIDTH_DEFAULT = .05 # trial time is split into non-overlapping bins with this width
THRESHOLD_FIRING_RATE_MIN = 1  # Lowest firing rate in Hz when filtering by firing rate
SPIKE_TRAIN_DILUTION_BOUND = .006 # spike train dilution lower bound
P_CRIT = .05  # p value for statistical signficance

# (minimum interspike interval when computin spike train cch)

ENUM_LICK_LEFT = 1
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

NEURON_UNIT_INFO_DTYPES = [ # data types corresponding to neuron unit info (for storing in numpy record array)
    ('unit_id', '<i4'),
    ('unit_quality', 'S10'),
    ('unit_x_in_um', '<f4'),
    ('depth_in_um', '<f4'),
    ('associated_electrode', '<i4'),
    ('shank', '<i4'),
    ('cell_type', 'S10'),  # one of 'Pyr' (Pyramidal Neurons), 'FS' (Fast Spiking), 'not classified', 'all' (all types)
    ('recording_location', 'S32')
]
