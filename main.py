import numpy as np
import pickle

fish_plane_ops = {
    0: ['02', '04', '06', '08', '10', '12', '14', '16', '18', '20'],  # can add '00'
    1: ['02', '04', '06', '08', '10', '12', '14', '16', '18', '20'],
    2: ['02', '04', '06', '08', '10', '12', '14', '16', '18', '20'],
    3: ['02', '04', '06', '08', '10', '12', '14', '16', '18', '20', '22'],
    4: ['01', '03', '05', '07', '09', '11'],  # , '13'], #, '15', '17', '19'],
    5: ['01', '03', '05', '07', '09', '11', '13', '15', '17', '19'],
    6: ['01', '03', '05', '07', '09', '11', '13', '15', '17', '19', '21'],
    7: ['01', '03', '05', '07', '09', '11', '13', '15'],
    8: ['01', '03', '05', '07', '09', '11', '13', '15', '17', '19', '21'],  # , '23'],
    9: ['00', '02', '04', '06', '08', '10', '12', '14', '16', '18', '20']
    }

if __name__ == '__main__':
    path = 'Data/df_data'  # where df_info and df_analysis are saved

    # dataframes with multi-index
    with open(path + 'df_info.pkl', 'rb') as pickle_in:
        df_info = pickle.load(pickle_in)  # idx = fish, plane

    with open(path + 'df_analysis.pkl', 'rb') as pickle_in:
        df_analysis = pickle.load(pickle_in)  # idx = fish, plane, cell

