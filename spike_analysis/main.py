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
    print(session_data_dict.keys())
    exit(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

