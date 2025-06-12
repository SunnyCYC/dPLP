# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:17:21 2025

@author: sunnycyc
"""

#%%
import os
import numpy as np
import tqdm.auto as tqdm
import json
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import glob
from pathlib import Path
import train_modules.utils as utils
import sys




#%%
def main():
    datasets = {
                    'gtzan':('./dataset/gtzan/', 
                            './dataset/gtzan/audio/', 
                            './dataset/gtzan/downbeats/')
                  }

        

    main_dst_dir = './dataset/'
    pp_method = 'LOC20'
    novelty_type = 'plp-curves'

    ### Settings for LOC20
    win_sec = 20 
    Fs_nov = 100
    global_height = 0.01
    pk_distance = 7
    # not_found = []

    for data_name, (dataset_dir, aud_dir, ref_dir) in datasets.items():
        print('==='*20)
        print('Dataset: {}'.format(data_name))
        print('==='*20)
        # break

        nov_folders = glob.glob(os.path.join(main_dst_dir, data_name, novelty_type, 
                                               "*"))
    
        
        for nov_folder in tqdm.tqdm(nov_folders):
            # break
            nov_type = os.path.basename(nov_folder)

           
                
                
            est_method = nov_type + '_GSMN'+'_' + pp_method
            est_dir = os.path.join(main_dst_dir, data_name, "beat_estimations_plpcompare", est_method)
            if not os.path.exists(est_dir):
                print('create folder:', est_dir)
                Path(est_dir).mkdir(parents = True, exist_ok= True)

            nov_paths = glob.glob(os.path.join(nov_folder, "*.npy"))
            nov_paths = sorted(nov_paths)
            for nov_path in tqdm.tqdm(nov_paths):
                # break
                est_path = os.path.join(est_dir, os.path.basename(nov_path).replace('.npy', '.beats'))
                if not os.path.exists(est_path):
                    nov = np.load(nov_path)
                    # print(nov.shape)                           
                    nov= gaussian_filter1d(nov, sigma=3)
                    nov = nov/nov.max()
                    # print(nov.shape)
                    
                    M = int(np.ceil(win_sec * Fs_nov))
                    locav = utils.compute_local_average(nov, M)
                    ## clip with global min height
                    locav = np.clip(locav, global_height, 1)
                    peaks, properties = find_peaks(nov, height = locav, 
                                           distance = pk_distance, )
                    peaks_sec = peaks/Fs_nov
                    np.savetxt(est_path, peaks_sec, fmt = '%.5f')
#%%
if __name__=='__main__':
    main()
