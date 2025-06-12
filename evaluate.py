# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 22:00:35 2025

@author: sunnycyc
"""

#%%
import os
import AnnotationCoverageRatio.acr_modules as acr 
from PredominantLocalPulseModules import tempo_modules as TempModules
import tqdm
import glob
import pandas as pd
import mir_eval.util as util
import numpy as np
import mir_eval
from pathlib import Path
import sys
import datetime

f_measure_threshold = 0.07




def renameLc(lc_res, lvalue):
    rename_lc_res = {}
    for k, v, in lc_res.items():
        # break
        rename_k = k.replace('-', '-L{}-'.format(lvalue))
        rename_lc_res[rename_k] = v
    return rename_lc_res
#%%
def main():
    csv_out_main_dir = './evaluation/'
    date = '2025-06-12'
    experiment_name = date+'_eval-dPLP'
    main_dst_dir = './dataset/' # location of estimations to evaluate
    
    # estimation_folder_type = 'beat-estimations-special'
    csv_out_folder = os.path.join(csv_out_main_dir, experiment_name, "csvfiles")
    
    # date ='2024-09-03_{}'.format(exp_foldername).replace('beat-estimations-pp-', '')
    if not os.path.exists(csv_out_folder):
        print("==="*20)
        print('Creating folder for csv files :{}'.format(csv_out_folder))
        # print('Method:{}'.format(exp_foldername))
        print("==="*20)
        Path(csv_out_folder).mkdir(parents = True, exist_ok = True)
    # Lvalues = [2, 3, 4]
    Lvalues = [2]

    dataset_dict = {
                    'gtzan':('./dataset/gtzan/', 
                            './dataset/gtzan/audio/', 
                            './dataset/gtzan/downbeats/')
                  }


    
    # pth_type = ['best', 'final', 'f1b', 'raw'][int(sys.argv[1])]
    pth_types = ['final', 'raw']
    # pth_types = ['raw']
    est_type = 'beat_estimations' 
    # songlevel_results = []

    for pth_type in pth_types:
        # break
        songlevel_results = []
        for dataset_name, (main_data_dir, audio_dir, beat_ann_dir) in dataset_dict.items():
            # break
            est_folders = glob.glob(os.path.join(main_dst_dir, dataset_name, est_type, 
                                                pth_type, "*"))

            
            for est_folder in tqdm.tqdm(est_folders):
                # break
                method = os.path.basename(est_folder)
                print("==="*27)
                print("Evaluation for Dataset: {}, Estimation: {}".format(dataset_name, method))
                print("==="*27) 
        
                ##########################################################
                # process all tracks for each estimation method
                ##########################################################
                if len(method.split('_')) == 5:
                    model_type = method.split('_')[0]
                    head_type = '_'.join(method.split('_')[1:3])
                elif len(method.split('_'))==4:
                    model_type, head_type = method.split('_')[:2]
                print('model type:{}, head:{}'.format(model_type, head_type))
                
                est_files = glob.glob(os.path.join(est_folder, "*.beats"))
                
                for est_file in tqdm.tqdm(est_files):
                    # break
                    
        
                    ###### get beat annotations for ACR
                    beat_annpath = os.path.join(beat_ann_dir, os.path.basename(est_file))
                    beat_ann = np.loadtxt(beat_annpath)
                    if len(beat_ann.shape)==2:
                        beat_ann = beat_ann[:, 0]

        
                    ###### exclude the measure number axis
                    est = np.loadtxt(est_file)
                    if len(est.shape)==2:
                        est = est[:, 0]
                    if est.shape==(): # with only one estimation
                        est = np.array([est])
                    ### trim 5 second
                    # ann = mir_eval.beat.trim_beats(ann)
                    # est = mir_eval.beat.trim_beats(est)
                    ###### main evaluation here:
                    F, P, R = mir_eval.onset.f_measure(beat_ann, est, window=f_measure_threshold)
                    # print('F: {:.3f}, P: {:.3f}, R: {:.3f}'.format(F, P, R))
                    result_dict = {
                        'Dataset': dataset_name, 
                        'model_type': model_type, 
                        'head_type': head_type,
                        'trackname': os.path.basename(est_file),
                        'Track': est_file, 
                        'F1': F, 
                        'R': R, 
                        'P': P, 
                        }

                    
                    for Lvalue in Lvalues:
                        # break
                        lc_res = TempModules.Lcorrect_eval(est, beat_ann, 
                                                      tolerance = f_measure_threshold, 
                                                      L = Lvalue)
                        result_dict.update(renameLc(lc_res, Lvalue))
                    

                    songlevel_results.append(result_dict)
        csv_spath = os.path.join(csv_out_folder, date + "-{}-dPLP-songlev-evalutation.csv".format(pth_type))
        df = pd.DataFrame(songlevel_results)
        df.to_csv(csv_spath)
#%%
if __name__=='__main__':
    main()

