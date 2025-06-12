# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 15:43:11 2025

@author: sunnycyc
"""

#%%
import os

os.sys.path.append('./')
import train_modules.dPLP_models_final as dPLP
import sys
import numpy as np
from numba import jit
import librosa
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn.functional as F
import torch.nn as nn

# sys.path.append('..')
import libfmp.b
import libfmp.c2
import libfmp.c6
from pathlib import Path
import tqdm.auto as tqdm
import glob
import train_modules.utils as utils

def compute_sinusoid_optimal(c, tempo, n, Fs, N, H):
    """Compute windowed sinusoid with optimal phase

    Notebook: C6/C6S2_TempogramFourier.ipynb
    sunny: remove the win part, as we'll add it when doing overlap-add for PLP

    Args:
        c (complex): Coefficient of tempogram (c=X(k,n))
        tempo (float): Tempo parameter corresponding to c (tempo=F_coef_BPM[k])
        n (int): Frame parameter of c
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size

    Returns:
        kernel (np.ndarray): Windowed sinusoid
        t_kernel (np.ndarray): Time axis (samples) of kernel
        t_kernel_sec (np.ndarray): Time axis (seconds) of kernel
    """
    # win = np.hanning(N)
    N_left = N // 2
    omega = (tempo / 60) / Fs
    t_0 = n * H
    t_1 = t_0 + N
    phase = - np.angle(c) / (2 * np.pi)
    t_kernel = np.arange(t_0, t_1)
    # kernel = win * np.cos(2 * np.pi * (t_kernel*omega - phase))
    kernel = np.cos(2 * np.pi * (t_kernel*omega - phase))
    t_kernel_sec = (t_kernel - N_left) / Fs
    return kernel, t_kernel, t_kernel_sec

def compute_sinusoid_optimal_weighted_sum(X, F_coef_BPM, n, Fs_nov, N, H, temp=1, plp_mode = 'softmax'):
    tempogram = np.abs(X)
    if plp_mode == 'softmax':
        coef_k = np.exp(tempogram[:, n] / temp) / np.exp(tempogram[:, n] / temp).sum()
    elif plp_mode =='equal':
        coef_k = np.ones(tempogram[:, n].shape[0])/tempogram.shape[0]

    kernel_sum = None

    tempo = (coef_k * F_coef_BPM).sum()
    corr = np.abs((coef_k * X[:, n]).sum())


    for k in range(F_coef_BPM.shape[0]):
        kernel, t_kernel, t_kernel_sec = compute_sinusoid_optimal(X[k,n], 
                                                F_coef_BPM[k], n, Fs_nov, N, H)
        
        if kernel_sum is None:
            kernel_sum = coef_k[k] * kernel
        else:
            kernel_sum += coef_k[k] * kernel

    return kernel_sum, t_kernel, t_kernel_sec, tempo, corr

def compute_plp(X, Fs, L, N, H, Theta, temp=1, plp_mode='softmax', return_nonrect=False):
    """Compute windowed sinusoid with optimal phase

    Notebook: C6/C6S3_PredominantLocalPulse.ipynb

    Args:
        X (np.ndarray): Fourier-based (complex-valued) tempogram
        Fs (scalar): Sampling rate
        L (int): Length of novelty curve
        N (int): Window length
        H (int): Hop size
        Theta (np.ndarray): Set of tempi (given in BPM)

    Returns:
        nov_PLP (np.ndarray): PLP function
    """
    win = np.hanning(N)
    ## normalization 
    win = win/(sum(win)/len(win))
    win = win/(len(win)/H)
    N_left = N // 2
    L_left = N_left
    L_right = N_left
    L_pad = L + L_left + L_right
    nov_PLP = np.zeros(L_pad)
    M = X.shape[1]
    tempogram = np.abs(X)
    for n in range(M):
        if plp_mode == 'argmax':
            k = np.argmax(tempogram[:, n])
            tempo = Theta[k]
            omega = (tempo / 60) / Fs
            c = X[k, n]
            phase = - np.angle(c) / (2 * np.pi)
            t_0 = n * H
            t_1 = t_0 + N
            t_kernel = np.arange(t_0, t_1)
            # kernel = win * np.cos(2 * np.pi * (t_kernel * omega - phase))
            kernel = np.cos(2 * np.pi * (t_kernel * omega - phase))
            
        elif plp_mode == 'softmax':
            F_coef_BPM = Theta
            Fs_nov = Fs
            kernel, t_kernel, t_kernel_sec, tempo, corr = compute_sinusoid_optimal_weighted_sum(X, 
                                                        F_coef_BPM, n, Fs_nov, N, H, temp)
        elif plp_mode == 'equal':
            F_coef_BPM = Theta
            Fs_nov = Fs
            kernel, t_kernel, t_kernel_sec, tempo, corr = compute_sinusoid_optimal_weighted_sum(X, 
                                                        F_coef_BPM, n, Fs_nov, N, H, temp, plp_mode = plp_mode)
        nov_PLP[t_kernel] = nov_PLP[t_kernel] + kernel*win
    nov_PLP = nov_PLP[L_left:L_pad-L_right]
    nov_PLP_rect = nov_PLP.copy()
    nov_PLP_rect[nov_PLP_rect < 0] = 0

    if return_nonrect:
        return nov_PLP_rect, nov_PLP
    else:
        return nov_PLP_rect

# nov_path = test_novpaths[0]
def genAPLP(nov_path, plp_type, fs_nov = 100, H = 10):
    theta_type, N_type = plp_type.split('-')
    if theta_type =='LN':
        Theta = np.arange(20, 320)
    elif theta_type == 'LG':
        Theta = np.logspace(np.log2(20), np.log2(320), 81, base = 2)
    N = int(N_type[1:])*fs_nov
    ### Load nov and get tempogram
    nov = np.load(nov_path)
    # print(nov.max(), nov.min())
    ## Apply Gaussian smoothing and max-norm
    nov= gaussian_filter1d(nov, sigma=3)
    nov = nov/nov.max()
    # print(nov.max(), nov.min())
    
    X_np, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs=fs_nov,
                                                                   N=N, H=H, Theta=Theta)

    plpnov = compute_plp(X_np, fs_nov, len(nov), N, H, Theta, plp_mode= 'argmax') 
    # print(plpnov.max(), plpnov.min())
    
    return plpnov

def genDPLP(nov_path, plp_type, fs_nov = 100, H = 10):
    theta_type, N_type = plp_type.split('-')
    if theta_type =='LN':
        Theta = np.arange(20, 320)
    elif theta_type == 'LG':
        Theta = np.logspace(np.log2(20), np.log2(320), 81, base = 2)
    N = int(N_type[1:])*fs_nov


    ### Load nov and get tempogram
    nov = np.load(nov_path)
    # print(nov.max(), nov.min())
    ## Apply Gaussian smoothing and max-norm
    nov= gaussian_filter1d(nov, sigma=3)
    nov = nov/nov.max()
    x = torch.tensor(nov)
    # print(nov.max(), nov.min())
    dplp = dPLP.Novelty2PLP(Fs_nov = fs_nov, N = N, H = H, 
                            Theta = torch.tensor(Theta), temp=1, plp_mode='softmax', )
    nov_dplp  = dplp(torch.tensor(x.unsqueeze(0)))
    dplp.eval()
    with torch.no_grad():
        nov_dplp_np = nov_dplp.detach().cpu().numpy()[0, :]
    
    return nov_dplp_np

            
#%%
def main():
    train_info_dir = './dataset/gtzan/train-info/'
    test_txt = os.path.join(train_info_dir, "test_files.txt")
    test_tracks = utils.getAudioPaths(test_txt)
    test_tbs = [os.path.basename(i).replace('.wav', '.npy') for i in test_tracks]

    main_dir = './dataset/gtzan/'
    out_dir = os.path.join(main_dir, 'plp-curves')
    settings = [f'{prefix}-{suffix}{i}' for i in [3, 5, 10] for prefix in \
                    ['LN', 'LG'] for suffix in ['A', 'S']]
    nov_src_dir = './dataset/gtzan/novelty/final/SFX/spectral_flux/'

    test_novpaths = [os.path.join(nov_src_dir, i) for i in test_tbs]
    for plp_type in tqdm.tqdm(settings):
        # break
        out_folder = os.path.join(out_dir, plp_type)
        if not os.path.exists(out_folder):
            print('create folder:', out_folder)
            Path(out_folder).mkdir(parents= True, exist_ok = True)
        for test_novpath in tqdm.tqdm(test_novpaths):
            # break
            out_path = os.path.join(out_folder, os.path.basename(test_novpath))
            if not os.path.exists(out_path):
                if plp_type.split('-')[1][0]=='A':
                    plpnov = genAPLP(test_novpath, plp_type, fs_nov = 100, H = 10)
                elif plp_type.split('-')[1][0]=='S':
                    plpnov = genDPLP(test_novpath, plp_type, fs_nov = 100, H = 10)
                else:
                    print('wrong plp_type:', plp_type)
                np.save(out_path, plpnov)

#%%

if __name__=='__main__':
    main()

