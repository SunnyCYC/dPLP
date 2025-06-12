# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:25:38 2025

@author: sunnycyc
"""

#%%
import os
import sys
import numpy as np
import train_modules.dPLP_models_final as dPLP
# from train_modules.beatdataset_30sec import AudioBeatTrack, AudioBeatDataset
import train_modules.utils as utils
import torch
import json
from pathlib import Path
import tqdm.auto as tqdm


def main():
    dataset_dirs = {
                    'gtzan':('./dataset/gtzan/', 
                            './dataset/gtzan/audio/', 
                            './dataset/gtzan/downbeats/')
                  }
    
        
    model_dirs = {
        'M1':'M1', 
        'M2':'M2', 
        'M3':'M3', 
        }
    
    exp_dir = './experiments/'
    main_dst_dir = './dataset/'
    
    cuda_num = int(sys.argv[1])
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    pth_type = 'final' 
    # pth_type = 'f1b'
    # pth_type = ['best', 'final', 'f1b', 'raw'][int(sys.argv[2])]
    # nov_type = 'max-beat-dbeat' # or #spectralnov
    print('device: {}, pth type:{}'.format(device, pth_type))
    
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
        Fs_nov = exp_setting['fs_nov']
        model_setting = json_data['model_setting']   

        if not modelname.startswith('M3'):
            start = int(exp_setting['theta'].split('_')[1])
            stop = int(exp_setting['theta'].split('_')[3])
            num = int(exp_setting['theta'].split('_')[5])
            Theta = torch.tensor(np.logspace(np.log2(start), np.log2(stop), num, base = 2) )
            model = dPLP.dPLPM1(Fs_nov = Fs_nov, Theta = Theta, **model_setting)
        else:
            model = dPLP.dPLPM3(Fs_nov = Fs_nov, **model_setting)
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
    
            assert os.path.exists(feature_dir), 'feature dir not exists:{}'.format(feature_dir)
            test_abts = utils.getABTtracks(audio_paths, feature_dir, ann_dir, SAMPLING_RATE, FEATURE_HOPSIZE)
 
            ### set output dir
            out_dir = os.path.join(main_data_dir, 'novelty', pth_type, modelname)
            for out_head in ['spectral_flux', 'fused']:
                out_folder = os.path.join(out_dir, out_head)
                if not os.path.exists(out_folder):
                    print('Creating folder: {}'.format(out_folder))
                    Path(out_folder).mkdir(parents = True, exist_ok = True)
            
            for test_abt in tqdm.tqdm(test_abts):
                # break
                nov_path = os.path.join(out_dir, 'spectral_flux', os.path.basename(test_abt.featurepath))
                fuse_path = os.path.join(out_dir, 'fused', os.path.basename(test_abt.featurepath))

                if not os.path.exists(nov_path):
                    run = True
                else: 
                    print('exists:', nov_path)

                if run:
                    feature_full, beat_label = test_abt.get_onehot_data(gsmn=None, seg_dur = 30)
                    # print(feature_full.shape, beat_label.shape)
                    x = torch.tensor(feature_full).unsqueeze(0)
                    yb = torch.tensor(beat_label)
                    # print(x.shape, yb.shape, )
                    
                    x, yb = x.to(device, torch.float32), yb.to(device)
                    out = model(x)
                    for k, v in out.items():
                        if k=='spectral_flux':
                            np.save(nov_path, v.detach().cpu().numpy()[0, :])
                        elif k=='fused_nov':
                            np.save(fuse_path,v.detach().cpu().numpy()[0, :])
                        elif k.startswith('N'):
                            plp_out_dir = os.path.join(out_dir, 'dplp-{}'.format(k))
                            if not os.path.exists(plp_out_dir):
                                Path(plp_out_dir).mkdir(parents = True, exist_ok= True)
                            plp_outpath = os.path.join(plp_out_dir, os.path.basename(test_abt.featurepath))
                            np.save(plp_outpath, v.detach().cpu().numpy()[0, :])
#%%
if __name__=="__main__":
    main()
