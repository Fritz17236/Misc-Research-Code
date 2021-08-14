# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from constants import DIR_RAW, DIR_SAVE
from containers import Session
import numpy as np
import scripts

np.seterr(all='raise')

if __name__ == '__main__':
    session_data_dict = scripts.load_all_session_data()
    regions = scripts.get_all_session_brain_regions(session_data_dict)
    print(regions)
    print(scripts.get_all_brain_region_pairs(regions))
    exit(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

