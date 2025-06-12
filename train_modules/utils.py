# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:57:40 2024

@author: sunnycyc
"""

#%%import librosa
# import shutil
import torch
import os
import numpy as np
from collections import defaultdict
# import torch
from torch.optim.optimizer import Optimizer
from .beatdataset_30sec import AudioBeatTrack, AudioBeatDataset

# from sklearn.utils.class_weight import compute_class_weight
# from .dbeatdataset import AudioBeatTrack, AudioBeatDataset
import tqdm
# import math

#  source code from : https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
class Lookahead(Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.

    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(1.0 - self.la_alpha, param_state['cached_params'])  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss

def save_checkpoint(
    state, is_best, path, target):
    # save full checkpoint including optimizer
    torch.save(
        state,
        os.path.join(path, target + '.chkpnt')
    )
    if is_best:
        # save just the weights
        torch.save(
            state['state_dict'],
            os.path.join(path, target + '.pth')
        )

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, best_loss = None):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = best_loss
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

def compute_local_average(x, M):
    """Compute local average of signal

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x (np.ndarray): Signal
        M (int): Determines size (2M+1) in samples of centric window  used for local average

    Returns:
        local_average (np.ndarray): Local average signal
    """
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average

def getAudioPaths(audio_txt):
    audiopaths = []
    with open(audio_txt, 'r') as file:
        for line in file.readlines():
            audiopaths.append(line.strip('\n'))
            
    return audiopaths

def getABTtracks(audiopaths, feature_dir, ann_dir, audio_rate, feature_hopsize):
    #######################################################################
    # for each audiopath, collect the corresponding feature and annotation, 
    # and init an AudioBeatTrack object
    #######################################################################
    audiobeattrack_list= []
    for audiopath in tqdm.tqdm(audiopaths):
        # break
        annpath = os.path.join(ann_dir, os.path.basename(audiopath).replace('.flac', '.beats').replace('.wav', '.beats'))
        if not os.path.exists(annpath):
            print("Ann does not exists: {}".format(annpath))
            # Ann does not exists: /rock_bring_the_noise.beats
        # else:
        #     annpaths.append(annpath)
        feature_path = os.path.join(feature_dir, os.path.basename(annpath).replace('.beats', '.npy'))
        # if not os.path.exists(feature_path):
        #     print("npy does not exists: {}".format(feature_path))
            # pass
        # else:
        #     featurepaths.append(feature_path)
        #######################################################################
        # init AudioBeatTrack
        #######################################################################
        if os.path.exists(annpath):
            audiobeattrack = AudioBeatTrack(audiopath, annpath, feature_dir, ann_dir, 
                                            audio_rate = audio_rate, 
                                            feature_hopsize = feature_hopsize )
            audiobeattrack_list.append(audiobeattrack)
    return audiobeattrack_list