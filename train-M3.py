# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:48:28 2025

@author: sunnycyc
"""

#%%
import os
import sys
sys.path.append('..')
from pathlib import Path
import glob
from train_modules import dPLP_models_final as dPLP
import librosa
from torchinfo import summary
import torch
import soundfile as sf
import numpy as np
import tqdm.auto as tqdm
from torch.utils.data import DataLoader
from train_modules.beatdataset_30sec import AudioBeatTrack, AudioBeatDataset
import train_modules.utils as utils
import time
import torch.nn as nn
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import mir_eval
import json
import torch

# from train_modules.utils import Lookahead
from torchinfo import summary
#### set up feature folder based on audio sampling rate, and feature hopsize
# NumPy seed
seed = 42
np.random.seed(seed)

# PyTorch seeds
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

# Ensure deterministic behavior (optional, for GPU)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

f_measure_threshold = 0.07
Fs_nov = 100
global_height = 0.01
pk_distance = 7
win_sec = 20

dataset_dirs = {
                'gtzan':('./dataset/gtzan/', 
                        './dataset/gtzan/audio/', 
                        './dataset/gtzan/downbeats/')
              }


def train(model, device, train_loader, optimizer, bce_weight):
    model.train()
    train_loss = {'all':0, 'F1':0, 'P':0, 'R':0,
                  # 'non-beat':0, 'd-beat':0, 
                  'beat': 0, 
                  'plpn3':0, 'plpn5':0, 'plpn10':0, 
                  'fuse':0,  
                  'len': len(train_loader.dataset)}
    pbar = tqdm.tqdm(train_loader, disable = False)
    for x, yb in pbar:
        # break
        pbar.set_description("Training batch")
        x,  yb = x.to(device, torch.float32), yb.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        beat_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([bce_weight]).to(device))( out['spectral_flux'].reshape((-1,)), yb.reshape((-1,)))
        fuse_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([bce_weight]).to(device))(out['fused_nov'].reshape((-1,)), yb.reshape((-1,)))
        
        loss = fuse_loss
        loss.backward()
        # Gradient clipping by value
        max_value = 0.5  # Max absolute value for gradients
        torch.nn.utils.clip_grad_value_(model.parameters(), max_value)
        #### save training loss
        train_loss['all'] += loss.detach().item()
        train_loss['beat'] += beat_loss.detach().item()
        train_loss['fuse'] += fuse_loss.detach().item()
        
        ### Beat tracking evaluation
        act = out['fused_nov'].detach().cpu().numpy()[0, :]
        nov= gaussian_filter1d(act, sigma=3)
        nov = nov/nov.max()
        M = int(np.ceil(win_sec* Fs_nov))
        locav = utils.compute_local_average(nov, M)
        ## clip with global min height
        locav = np.clip(locav, global_height, 1)
        peaks, properties = find_peaks(nov, height = locav, 
                               distance = pk_distance, )
        peaks_est = peaks/Fs_nov
        ### annotations
        yb_np = yb[0, :, 0].detach().cpu().numpy()
        locav_ann = utils.compute_local_average(yb_np, M)
        ## clip with global min height
        locav_ann = np.clip(locav_ann, global_height, 1)
        peaks_ann, properties = find_peaks(yb_np, height = locav_ann, 
                               distance = pk_distance, )
        beat_ann = peaks_ann/Fs_nov
        F, P, R = mir_eval.onset.f_measure(beat_ann, peaks_est, window=f_measure_threshold)
        # print('F:{:.3f}, P:{:.3f}, R:{:.3f}'.format(F, P, R))
        #### save training loss

        train_loss['F1'] += F/len(pbar)
        train_loss['P'] += P/len(pbar)
        train_loss['R'] += R/len(pbar)

        
        optimizer.step()

    return train_loss

def valid(model, device, valid_loader, bce_weight ):
    model.eval()
    valid_loss =  {'all':0, 'F1':0, 'P':0, 'R':0,
                  # 'non-beat':0, 'd-beat':0, 
                  'beat': 0, 
                  'plpn3':0, 'plpn5':0, 'plpn10':0, 
                  'fuse':0,  
                  'len': len(valid_loader.dataset)}
    pbar = tqdm.tqdm(valid_loader, disable = False)
    with torch.no_grad():
        for x, yb in pbar:
            pbar.set_description("Valid batch")
            x, yb = x.to(device, torch.float32), yb.to(device)

            out = model(x)
            beat_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([bce_weight]).to(device))( out['spectral_flux'].reshape((-1,)), yb.reshape((-1,)))
            fuse_loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([bce_weight]).to(device))(out['fused_nov'].reshape((-1,)), yb.reshape((-1,)))

            loss = fuse_loss
            #### save training loss
            valid_loss['all'] += loss.detach().item()
            valid_loss['beat'] += beat_loss.detach().item()
            valid_loss['fuse'] += fuse_loss.detach().item()
            
            ### Beat tracking evaluation
            act = out['fused_nov'].detach().cpu().numpy()[0, :]
            nov= gaussian_filter1d(act, sigma=3)
            nov = nov/nov.max()
            M = int(np.ceil(win_sec* Fs_nov))
            locav = utils.compute_local_average(nov, M)
            ## clip with global min height
            locav = np.clip(locav, global_height, 1)
            peaks, properties = find_peaks(nov, height = locav, 
                                   distance = pk_distance, )
            peaks_est = peaks/Fs_nov
            ### annotations
            yb_np = yb[0, :, 0].detach().cpu().numpy()
            locav_ann = utils.compute_local_average(yb_np, M)
            ## clip with global min height
            locav_ann = np.clip(locav_ann, global_height, 1)
            peaks_ann, properties = find_peaks(yb_np, height = locav_ann, 
                                   distance = pk_distance, )
            beat_ann = peaks_ann/Fs_nov
            F, P, R = mir_eval.onset.f_measure(beat_ann, peaks_est, window=f_measure_threshold)
            # print('F:{:.3f}, P:{:.3f}, R:{:.3f}'.format(F, P, R))
            #### save training loss

            valid_loss['F1'] += F/len(pbar)
            valid_loss['P'] += P/len(pbar)
            valid_loss['R'] += R/len(pbar)
            
    return valid_loss



#%%
# pretreined_dir = '/media/HDDStorage/sunnycyc/scripts/dual-domain-beat-tracking/experiments-blstm/'
def main():

    continue_from_old = False
    # Theta = torch.tensor(np.logspace(np.log2(start), np.log2(stop), num, base = 2) )
    exp_setting = {
        'pretrained': False, 
        'sampling-rate': 44100, 
        'feature-hopsize': 441, 
        'feature-folder': 'stft_spectrograms',
        'segment-duration': 30 , #seconds
        'train-max-epoch': 300,     
        'learning-rate': 0.1, 
        'patience': 5, 
        'optimizer': 'Lookahead-Adam', 
        'feature_size' : 314,
        'fs_nov': Fs_nov,
        'loss-type': 'BCE',
        'target-gsmn': True,
        'bce_weight' : 3,
        'batch_size' : 8,
        }
    model_setting = {
        'N_bands':8, 
        'gamma_init':10, 
        'N_differentiation':5, 
        'a_lrelu':0.0, 
        'N_local_average':11,
        'N_gaussian': 15, 
        'sigma_gaussian': 3,
        'gamma_trainable': True, 
        'diff_trainable' : True,  
        'loc_avg_trainable': False,
        'weighted_sum_trainable': True, 
        'gaussian_trainable': False,
        'fuser_fc_trainable': True, 
        'fuser_gs_trainable' : True, 
        'fuser_N_gaussian' : 15, 
        'fuser_sigma_gaussian' : 3,
        'H': 10,
        }
    



    # Feature_type = exp_setting['feature-type'] ## may also be beatNet, or madmomRNNDB, ....
    patience = exp_setting['patience']
    train_epochs = exp_setting['train-max-epoch']
    main_dir = './experiments/'
    
    cuda_num = int(sys.argv[1])
    # cuda_num = 1
    cuda_str = 'cuda:'+str(cuda_num)
    device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu') 

    # must assign
    lr = exp_setting['learning-rate']
    exp_name = 'M3'  #+str(fold_num) + '_v{}'.format( vnum)
    exp_dir = os.path.join(main_dir, exp_name)
    target_jsonpath = exp_dir
    exp_setting['experiment-folder'] = exp_dir
    exp_setting['main-exe-dir'] = main_dir


    #######################################################################
    ### collect audiobeattrack objects from each dataset
    #######################################################################

    train_abt_list = []
    valid_abt_list = []
    for dname, (dataset_dir, audio_dir, ann_dir) in dataset_dirs.items():
        # print(dname, '\n', dataset_dir, '\n', audio_dir, f_end, '\n', ann_dir,'\n' )
        # break
        print("==="*20)
        print("Processing {} dataset...".format(dname))
        print("==="*20)
        #######################################################################
        ### collect audiopaths based on audio_files.txt in each audio_dir
        #######################################################################       
        train_txt = os.path.join(dataset_dir, 'train-info', '{}_files.txt'.format('train'))
        train_audiopaths = utils.getAudioPaths(train_txt)
        valid_txt = os.path.join(dataset_dir, 'train-info', '{}_files.txt'.format('valid'))
        valid_audiopaths = utils.getAudioPaths(valid_txt)

        #######################################################################
        ### setup folder directory for beat features, and collect featurepaths
        #######################################################################
        feature_location = os.path.join(dataset_dir,  exp_setting['feature-folder'])


        train_abts = utils.getABTtracks(train_audiopaths, feature_location, ann_dir, 
                                        exp_setting['sampling-rate'], exp_setting['feature-hopsize'])
        valid_abts = utils.getABTtracks(valid_audiopaths, feature_location, ann_dir, 
                                        exp_setting['sampling-rate'], exp_setting['feature-hopsize'])

        print('---'*20)
        print('train_tracks: {}, valid_tracks: {}'.format(len(train_abts), len(valid_abts)))
        print('---'*20)
        train_abt_list += train_abts
        valid_abt_list += valid_abts
    #######################################################################
    ### init train/valid dataloaders
    #######################################################################
    trainset = utils.AudioBeatDataset(train_abt_list, segment_dur_sec = exp_setting['segment-duration'], 
                                gsmn_label = exp_setting['target-gsmn'],
                                freq_dim = 1024,
                                ) 
    train_loader = DataLoader( trainset, batch_size = exp_setting['batch_size'], shuffle = True)
    validset = utils.AudioBeatDataset(valid_abt_list, segment_dur_sec = exp_setting['segment-duration'], 
                                freq_dim = 1024,
                                gsmn_label = exp_setting['target-gsmn'],
                                )
    valid_loader = DataLoader(validset, batch_size = exp_setting['batch_size'], shuffle = True)

    #######################################################################
    ### information to save
    #######################################################################
    if not os.path.exists(exp_dir):
        print('Creating folder:{}'.format(exp_dir))
        Path(exp_dir).mkdir(parents = True, exist_ok = True)

    model_type = 'M3'
    exp_setting['model_type'] = model_type

    model = dPLP.dPLPM3(Fs_nov = Fs_nov, **model_setting)
    if continue_from_old:
        print('Continue training from old final model:', exp_dir)
        model_path = os.path.join(exp_dir, "model.chkpnt")
        model.load_state_dict(torch.load(model_path, map_location = device)['state_dict'])
        
    elif exp_setting['pretrained']:
        # print('load pretreind rnn model...')
        print('not implemented!!!!!')
        # rnn_name = '2024-09-11-BLSTM-BDB_f{}_fulltrack_v0'.format(fold_num)
        # rnn_path = os.path.join(pretrained_dir, rnn_name, 'model.pth')
        # model.NovNet.load_state_dict(torch.load(rnn_path, map_location = device))
        # summary( model)


    model.to(device)
    # summary( model)
    # print(model.dTempogram.window is model.dPLP.win)  # Should print True if shared
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Trainable Parameter: {name}, Shape: {param.shape}")
    #%%
    lr = exp_setting['learning-rate']
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay= 0.00001
        )

    optimizer =  utils.Lookahead(optimizer)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.3, 
            patience=80,
            cooldown=10
        )

 
    es = utils.EarlyStopping(patience= patience)
    if continue_from_old:
        json_path = os.path.join(exp_dir, 'model.json')
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
        start_epoch = json_data["epochs_trained"]+1
        lr_change_epoch = json_data["lr_change_epoch"]
        stop_t = json_data["stop_t"]
        train_times = json_data["train_time_history"]
        best_epoch = json_data["best_epoch"]
        best_f1_epoch = json_data["best_f1_epoch"]
        best_f1 = json_data["best_f1"]
        
        train_times = json_data['train_time_history']
        train_losses = json_data['train_loss_history']
        train_beat_losses = json_data['train_beat_loss_history']
        train_plpn3_losses = json_data['train_plpn3_loss_history']
        train_plpn5_losses = json_data['train_plpn5_loss_history']
        train_plpn10_losses = json_data['train_plpn10_loss_history']
        train_fuse_losses = json_data['train_fuse_loss_history']
        ### valid losses
        valid_losses = json_data['valid_loss_history']
        valid_beat_losses = json_data['valid_beat_loss_history']
        valid_plpn3_losses = json_data['valid_plpn3_loss_history']
        valid_plpn5_losses = json_data['valid_plpn5_loss_history']
        valid_plpn10_losses = json_data['valid_plpn10_loss_history']
        valid_fuse_losses = json_data['valid_fuse_loss_history']
    else:
        start_epoch = 1
        best_f1_epoch = 0
        best_f1 = 0
        lr_change_epoch = []
        stop_t = 0
        best_epoch = 0
        train_times = []
        train_losses = []
        train_beat_losses = []
        train_plpn3_losses = []
        train_plpn5_losses = []
        train_plpn10_losses = []
        train_fuse_losses = []
        train_F1s = []
        train_Ps = []
        train_Rs = []
        ### valid losses
        valid_losses = []
        valid_beat_losses = []
        valid_plpn3_losses = []
        valid_plpn5_losses = []
        valid_plpn10_losses = []
        valid_fuse_losses = []
        valid_F1s = []
        valid_Ps = []
        valid_Rs = []

    t = tqdm.trange(start_epoch, train_epochs +1, disable = False)

    ### time information
    for epoch in t:
        # break
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = train(model, device, train_loader, optimizer, 
                           bce_weight= exp_setting['bce_weight'])
        valid_loss = valid(model, device, valid_loader, 
                           bce_weight= exp_setting['bce_weight'])

        scheduler.step(valid_loss['all']/valid_loss['len'])
        ### train losses
        train_losses.append(train_loss['all']/train_loss['len'])
        train_beat_losses.append(train_loss['beat']/train_loss['len'])
        train_fuse_losses.append(train_loss['fuse']/train_loss['len'])
        train_F1s.append(train_loss['F1'])
        train_Ps.append(train_loss['P'])
        train_Rs.append(train_loss['R'])
        ### valid losses
        valid_losses.append(valid_loss['all']/valid_loss['len'])
        valid_beat_losses.append(valid_loss['beat']/valid_loss['len'])
        valid_fuse_losses.append(valid_loss['fuse']/valid_loss['len'])
        valid_F1s.append(valid_loss['F1'])
        valid_Ps.append(valid_loss['P'])
        valid_Rs.append(valid_loss['R'])
        if valid_loss['F1']> best_f1:
            best_f1 = valid_loss['F1']
            best_f1_epoch = epoch 
            utils.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'best_loss': es.best,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    },
                    is_best=True,
                    path=exp_dir,
                    target='model_bestF1'
                )


        t.set_postfix(
        train_loss=train_loss['all']/train_loss['len'], 
        val_loss=valid_loss['all']/valid_loss['len']
        )

        stop = es.step(valid_loss['all']/valid_loss['len'])

        if valid_loss['all']/valid_loss['len'] == es.best:
            best_epoch = epoch

        utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': es.best,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                },
                is_best=valid_loss['all']/valid_loss['len'] == es.best,
                path=exp_dir,
                target='model'
            )

            # save params
        params = {
                'epochs_trained': epoch,

                'best_loss': es.best,
                'best_epoch': best_epoch,
                'best_f1': best_f1, 
                'best_f1_epoch': best_f1_epoch,
                'train_loss_history': train_losses,
                'train_beat_loss_history': train_beat_losses, 
                'train_plpn3_loss_history': train_plpn3_losses, 
                'train_plpn5_loss_history': train_plpn5_losses, 
                'train_plpn10_loss_history': train_plpn10_losses, 
                'train_fuse_loss_history': train_fuse_losses,
                'train_F1_history': train_F1s,
                'train_P_history': train_Ps,
                'train_R_history': train_Rs,
                'valid_loss_history': valid_losses,
                'valid_beat_loss_history': valid_beat_losses, 
                'valid_F1s_history': valid_F1s,
                'valid_P_history': valid_Ps,
                'valid_R_history': valid_Rs,
                'valid_plpn3_loss_history': valid_plpn3_losses, 
                'valid_plpn5_loss_history': valid_plpn5_losses,
                'valid_plpn10_loss_history': valid_plpn10_losses,
                'valid_fuse_loss_history': valid_fuse_losses,
                'train_time_history': train_times,
                'num_bad_epochs': es.num_bad_epochs,
                'lr_change_epoch': lr_change_epoch,
                'stop_t': stop_t,
                'exp_setting': exp_setting,
                'model_setting': model_setting,
            }

        with open(os.path.join(target_jsonpath,  'model' + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping and retrain")
            stop_t +=1
            if stop_t >=5:
                break
            lr = lr*0.2
            lr_change_epoch.append(epoch)
            optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay= 0.00001
                )

            optimizer = utils.Lookahead(optimizer)

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=0.3, 
                    patience=80,
                    cooldown=10
                )

            es = utils.EarlyStopping(patience= patience, best_loss = es.best)
            
            
#%%
if __name__=='__main__':
    main()