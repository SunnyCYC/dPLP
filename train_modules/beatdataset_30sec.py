# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:23:13 2023


@author: Sunny
"""

#%%
import os
from pathlib import Path
from torch.utils.data import Dataset
import tqdm
import numpy as np
import librosa
# import glob
# from scipy.ndimage import maximum_filter1d
from scipy.ndimage import gaussian_filter1d



### Settings
SAMPLING_RATE = 44100
MONO = True
DTYPE = np.float32
RANDOM_SEED = 100
np.random.seed(RANDOM_SEED)


def beat2spec_3D(beats, spec_timesteps = 1000, sr = 44100, hop_length = 441):
    """ purpose:
        1. convert beat information to shape of spectrogram 
        2. save beat, downbeat, nonbeat info to dimension 2, 1, 0 """
        
    beat_label = np.zeros((spec_timesteps, 1))
    dbeat_label = np.zeros((spec_timesteps, 1))
    nonbeat_label = np.ones((spec_timesteps, 1))
    for (beat_time, beat_type) in beats:
        # break
        time_ind = int(beat_time*sr/hop_length)
        if time_ind >= spec_timesteps: ### to exclude case of gtzan with beat ann longer than audio duration
            continue 
        if str(int(beat_type)) == '1':
            dbeat_label[time_ind, 0] = 1
        
        beat_label[time_ind, 0] = 1
        nonbeat_label[time_ind, 0] = 0

    return nonbeat_label, dbeat_label, beat_label


def GSMN_Label(beat_frames, sigma = 3):
    beat_frames = gaussian_filter1d(beat_frames[:, 0], sigma = 3)
    beat_frames = beat_frames/beat_frames.max()
    # print('smoothed shape:', beat_frames.shape)
    return beat_frames[:, np.newaxis]
                                  

class AudioBeatDataset(Dataset):
    
    
    def __init__(self, 
                 audiobeattrack_list,
                 segment_dur_sec = 30, # duration for training
                 freq_dim = 1024, # size of freq dim of stft
                 gsmn_label = False
                 ):

        self.audiobeattrack_list = audiobeattrack_list
        self.segment_dur_sec = segment_dur_sec
        self.gsmn_label = gsmn_label
        self.freq_dim = freq_dim

        
    def __len__(self):
        return len(self.audiobeattrack_list)
    
    def __getitem__(self, index):
        audiobeattrack = self.audiobeattrack_list[index]
        
        return audiobeattrack.get_onehot_data(gsmn = self.gsmn_label, 
                                              seg_dur = self.segment_dur_sec, 
                                              freq_dim = self.freq_dim)
    
    def __add__(self, other):
        return ConcatAudioBeatDataset([self, other])
    
    def precompute(self, ):
        print("===*20")
        print("Precomputing features...")
        print("===*20")
        for audiobeattrack in tqdm.tqdm(self.audiobeattrack_list):
            audiobeattrack.precompute_stft()

class AudioBeatTrack(object):
    
    def __init__(self, 
                audiopath, 
                annpath,
                feature_folder, 
                ann_folder,
                audio_rate = SAMPLING_RATE, 
                feature_hopsize = 441, 

                ):
        self.audiopath = audiopath
        self.annpath = annpath
        self.feature_folder = feature_folder
        self.ann_folder = ann_folder
        self.audio_rate = audio_rate
        self.feature_hopsize = feature_hopsize
        self.featurepath = os.path.join(self.feature_folder, 
                                        os.path.basename(self.annpath).replace('.beats', '.npy'))
        self.feature_fps = self.audio_rate/self.feature_hopsize
        
    
    def get_audio(self):
        audio, rate = librosa.load(self.audiopath, sr = self.audio_rate, mono = MONO, dtype = DTYPE)
        return audio
    
    def get_beatann(self):
        beat_ann_full = np.loadtxt(self.annpath)

        return beat_ann_full
    
    def precompute_stft(self):
        if os.path.exists(self.featurepath):
            print('exists:{}'.format(os.path.basename(self.featurepath)))
        else:
            if not os.path.exists(self.feature_folder):
                print('Created feature folder: {}'.format(self.feature_folder))
                Path(self.feature_folder).mkdir(parents = True, exist_ok = True)
            ##### calculate madmom feature #####
            
           
            # feature_full = madmom_preproc(self.audiopath)
            audio = self.get_audio()
            stft = librosa.stft(audio, n_fft=2048, hop_length=441, win_length=1024, window='hann')
            np.save(self.featurepath, np.abs(stft))
        
    
    def get_feature(self):
        if not os.path.exists(self.featurepath):
            self.precompute_stft()
            feature_full = np.load(self.featurepath)
        else:
            feature_full = np.load(self.featurepath)
        
        
        return feature_full
    
    def get_onehot_data(self, gsmn = False, seg_dur = 30, freq_dim = 1024):
        feature_full = self.get_feature()
        beat_ann_full = self.get_beatann()
        # print('feat shape:{}, beat shape:{}'.format(feature_full.shape, beat_ann_full.shape))
        ### Use only the first 30 seconds
        end_frame = int(self.feature_fps*seg_dur)
        if feature_full.shape[1]<end_frame:
            # print('feat shape:{}, beat shape:{}'.format(feature_full.shape, beat_ann_full.shape))
            feature_out = np.zeros((freq_dim, end_frame))
            feature_out[:freq_dim, :feature_full.shape[1]] = feature_full[:freq_dim]
            # print('feat out shape:{}'.format(feature_out.shape))
        else:
            feature_out = feature_full[:freq_dim, :end_frame]
        ann_ids = np.where(beat_ann_full[:, 0]<seg_dur)[0]
        beat_ann_out = beat_ann_full[ann_ids, :]
        
        # ### only implement full track version
        nonbeat_label, dbeat_label, beat_label = beat2spec_3D(beat_ann_out, 
                                              spec_timesteps = feature_out.shape[1], 
                                  sr = self.audio_rate, hop_length = self.feature_hopsize)
        if not gsmn:
            
            return feature_out, beat_label
        else:
            # print('label shape:', dbeat_label.shape)
            return feature_out, GSMN_Label(beat_label)
                
class ConcatAudioBeatDataset(AudioBeatDataset):
    r"""Concatenate multiple `AudioBeatDataset`s.

    Arguments:
        datasets (list): A list of `AudioBeatsDataset` objects.
    """

    def __init__(self, datasets):
           

        audiobeattrack_list = []
        for dataset in datasets:
            audiobeattrack_list += dataset.audiobeattrack_list

        super().__init__(audiobeattrack_list)
        

