# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:25:38 2025

@author: sunnycyc
"""

#%%
import os
os.sys.path.append('./')
import sys
import numpy as np
import train_modules.dPLP_models_final as dPLP
# from train_modules.beatdataset_30sec import AudioBeatTrack, AudioBeatDataset
import torch
import train_modules.utils as utils

import json
from pathlib import Path
import tqdm.auto as tqdm



dataset_dirs = {
                'gtzan':('./dataset/gtzan/', 
                        './dataset/gtzan/audio/', 
                        './dataset/gtzan/downbeats/')
              }

    
model_dirs = {
    'SFX':'SFX', 
    }

def main():

    exp_dir = './experiments/'
    main_dst_dir = './dataset/'
    

    cuda_num = int(sys.argv[1])
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # pth_type = 'best' # or 'final'
    # pth_types = ['best', 'final', 'f1b', 'raw']
    pth_types = ['final', 'raw']
    # pth_type = 'final'

    print('device: {}'.format(device))
    for pth_type in pth_types:
        # break
        for modelname, modelfolder in model_dirs.items():
            # break
            print("==="*20)
            print('Model name: {}'.format(modelname))
            print('folder: {}, Pth type: {}'.format(modelfolder, pth_type))
            print("==="*20)
            exp_folder = os.path.join(exp_dir, modelfolder)
            json_path = os.path.join(exp_folder, 'model.json')
            with open(json_path, 'r') as json_file:
                json_data = json.load(json_file)

            exp_setting = json_data['exp_setting']
            SAMPLING_RATE = exp_setting['sampling-rate']
            FEATURE_HOPSIZE = exp_setting['feature-hopsize'] ### use madmom setting first            
            model_setting = json_data['model_setting']
            
            model = dPLP.SpectralFlux(**model_setting)
            # summary( model)
            if pth_type == 'best':
                model_path = os.path.join(exp_folder, "model.pth")
                model.load_state_dict(torch.load(model_path))
            elif pth_type == 'final':
                model_path = os.path.join(exp_folder, "model.chkpnt")
                model.load_state_dict(torch.load(model_path)['state_dict'])
            elif pth_type == 'f1b':
                model_path = os.path.join(exp_folder, "model_bestF1.chkpnt")
                model.load_state_dict(torch.load(model_path)['state_dict'])
            elif pth_type == 'raw': # non-trained models
                pass
            model.to(device)
            model.eval() 
            
            for dname, (dataset_dir, audio_dir, ann_dir) in dataset_dirs.items():
                # break
                print("==="*20)
                print("Processing {} dataset...".format(dname))
                print("==="*20)
                main_data_dir = os.path.join(main_dst_dir, dname)
                audio_txt = os.path.join(audio_dir, "audio_files.txt")
                audio_paths = utils.getAudioPaths(audio_txt)
                
                feature_dir = os.path.join(dataset_dir,  exp_setting ['feature-folder'])
                test_abts = utils.getABTtracks(audio_paths, feature_dir, ann_dir, SAMPLING_RATE, FEATURE_HOPSIZE)
                
                out_dir = os.path.join(main_data_dir, 'novelty', pth_type, modelname)
                for out_type in ['spectral_flux']:
                    outfolder = os.path.join(out_dir, out_type)
                    if not os.path.exists(outfolder):
                        print('Creating folder: {}'.format(outfolder))
                        Path(outfolder).mkdir(parents = True, exist_ok = True)
                
                for test_abt in tqdm.tqdm(test_abts):
                    # break
                    nov_path = os.path.join(out_dir, 'spectral_flux', os.path.basename(test_abt.featurepath))
                    outpaths = [nov_path]
                    run = any([not os.path.exists(i) for i in outpaths])
                    if run:
                        feature_full, beat_label = test_abt.get_onehot_data(gsmn=None)
                        # print(feature_full.shape, nonbeat_label.shape, dbeat_label.shape, beat_label.shape)
                        x = torch.tensor(feature_full).unsqueeze(0)
                        yb = torch.tensor(beat_label)
                        # print(x.shape, ynb.shape, yb.shape, ydb.shape)
                        
                        x, yb = x.to(device, torch.float32), yb.to(device)
                        out = model(x)
                        np.save(nov_path, out['x_act'].detach().cpu().numpy()[0, :])
                    
            
#%%
if __name__ =='__main__':
    main()
            
