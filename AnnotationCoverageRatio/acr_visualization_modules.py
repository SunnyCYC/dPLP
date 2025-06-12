# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 19:40:18 2024

@author: sunnycyc
"""

#%%
import os
# os.chdir('/media/HDDStorage/sunnycyc/scripts/downbeat-experiments/')
os.chdir('/home/ALABSAD/sunnycyc/JupyterIPNB/daga-downbeat/')
from AnnotationCoverageRatio import acr_modules as acr_module
import musicalTimeModules as MTM
# from PredominantLocalPulseModules import modules as PPTModules
# from PredominantLocalPulseModules import tempo_modules as TempModules
import tqdm
import glob
import pandas as pd
import mir_eval.util as util
# import librosa
# import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# import soundfile as sf
# import madmom.features.beats as mdm
# from madmom.features.onsets import CNNOnsetProcessor, peak_picking
# from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
# from scipy.signal import find_peaks # sppk
import libfmp.c6 as libfmp
import mir_eval
from pathlib import Path


# Beat_FPS = 100

y_tick_dict ={
    'onbeat':'onbeat', 
    'offbeat':'offbeat', 
    'double':'double', 
    'triple':'triple', 
    'quadruple':'quadruple',  
    'half':'half', 
    'third':'third', 
    'quarter': 'quarter', 
    'any': 'any'} 

c_types=['onbeat', 'offbeat', 'half', 'third',  'quarter',
         'double', 'triple', 'quadruple', 
         'any']
c_ticks = {y_tick_dict[y]:ind for ind, y in enumerate(c_types)}


def fullTrackACR(beat_est, beat_ann, tolerance = 0.07, L=2, fontsize = 12, title = None,
                 half_offbeat = True, double= True, half= True, 
                 third_offbeat = True, triple= True, third=True, 
                 quadruple = True,
                 return_dict = False, quarter = True,
                 return_cframe = True, FPS = 100, 
                 musical_time = False, sync_csvpath = None, measure_decimal = 1):
    acr_res = acr_module.anyMetLev_eval(beat_est, beat_ann, tolerance = tolerance, L =L,
                      half_offbeat = half_offbeat, 
                      double= double, half = half, 
                      third_offbeat = third_offbeat, 
                      triple = triple, third = third, 
                      quadruple = quadruple,
                      return_dict = return_dict, quarter = quarter,
                      return_cframe = return_cframe, FPS = FPS)
    c_list = []
    for c_type in acr_module.c_types:
        c_list.append(acr_res['correct_frame_results'][c_type][:, np.newaxis])
    c_array = np.array(c_list).squeeze()
    # c_dicts = {est_type: c_array}
    #########################
    # ACR -- Overview Plot
    #########################
    # if est_type in ppts_2overview:
    
    fig, ax = plt.subplots(1, 1, figsize = (7, 2))
    ax.imshow(c_array, aspect = 'auto', cmap = 'Greens', interpolation='none')
    ax.set_yticks(range(len(c_ticks)) )
    ax.grid(axis='y', linewidth = 1)
    ax.set_yticklabels(c_ticks, fontsize = fontsize)
    ax.tick_params(axis = 'y', rotation = 30)
    ax.get_yaxis().set_label_coords(-0.2,0.5)
    if title:
        ax.set_title(title)
    xticks = np.arange(0, int(len(c_array.T)), 5000)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks/FPS)
    ax.set_xlabel('Time (seconds)')
    if musical_time == True:
        if not os.path.exists(sync_csvpath):
            print('Synchronization information required for Musical Time axis.')
            print('Not exists: given sync csvpath:{}'.format(sync_csvpath))
        else:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(xticks)
            xticks_sec = xticks/FPS
            xticks_measure = [np.round(MTM.queryMusicaltime(in_sec, sync_csvpath), measure_decimal) for in_sec in xticks_sec]
            ax2.set_xticklabels(xticks_measure)
            ax2.set_xlabel('Musical Time (measure number)')
        # second_ax = ax.secondary_xaxis('top', functions = (MTM.queryMusicaltime( sync_csvpath = sync_csvpath), 
        #                                                    MTM.queryPhysicaltime( sync_csvpath = sync_csvpath)))
    
    return fig
#%%
# for in_sec in xticks_sec:
#     print(in_sec)
#%%
# xticks_sec = xticks/FPS
# sync_csvpath = csvpath

# xticks_measure = [np.round(MTM.queryMusicaltime(in_sec, sync_csvpath), 2) for in_sec in xticks_sec]
# MTM.queryMusicaltime(xticks_sec, sync_csvpath = sync_csvpath)


# beat_est = b_ann
# beat_ann = db_ann
# L =2
# tolerance = 0.07
# title = None
# half_offbeat = True
# double= True
# half= True

# third_offbeat = True 
# triple= True
# third=True
# quadruple = True
# return_dict = False
# quarter = True
# return_cframe = True
# FPS = 100

#%% test region
# sync_csvpath = csvpath

# beat_ann_dir = '/media/HDDStorage/sunnycyc/datasets/Chopin_FiveMazurkas/downbeats/'
# db_annpath = '/media/HDDStorage/sunnycyc/datasets/Chopin_FiveMazurkas/annotations/downbeats/Chopin_Op017No4_Ohlsson-1999_pid9153-13.beats'
# b_annpath = os.path.join(beat_ann_dir, os.path.basename(db_annpath))

# db_ann = np.loadtxt(db_annpath)
# if len(db_ann.shape)==2:
#     db_ann = db_ann[:, 0]
# b_ann = np.loadtxt(b_annpath)
# if len(b_ann.shape)==2:
#     b_ann = b_ann[:, 0]
    
# fig = fullTrackACR(b_ann, db_ann, title = 'Original annotation', fontsize = 12, 
#                    musical_time= True, sync_csvpath = csvpath)
#%% query physical time test
# csvpath = '/home/ALABSAD/sunnycyc/GroupMM/AL_Work/WorkCC/datasets/BeethovenSonatasFull/ann_audio_syncInfo/Beethoven_Op110-01_FG67.csv'
# test_measure = [0, 10, 2.22, 10100]
# for in_measure in test_measure:
#     print('------'*3)
#     out_sec = MTM.queryPhysicaltime(in_measure, csvpath)
#     print('measure:{}, sec:{}'.format(in_measure, out_sec))
