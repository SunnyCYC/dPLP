# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:42:29 2025

@author: sunnycyc
"""

#%%
import os
# os.chdir('/media/HDDStorage/sunnycyc/scripts/downbeat-experiments/')
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import libfmp.b
import libfmp.c2
import libfmp.c6
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import IPython.display as ipd
import pickle
from scipy import signal
from scipy.interpolate import interp1d
import glob
import pandas as pd
import librosa
import mir_eval
import mir_eval.util as util
colors = list(mcolors.TABLEAU_COLORS.keys())

def softmax(x, dim=0):
    return np.exp(x) / np.exp(x).sum(axis=dim, keepdims=True)  


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
    # coef_k_soft = np.exp(tempogram[:, n] / temp) / np.exp(tempogram[:, n] / temp).sum()
    # coef_k_equal = np.ones(tempogram[:, n].shape[0])/tempogram.shape[0]
    kernel_sum = None

    # tempo = (coef_k_soft * F_coef_BPM).sum()
    # corr = np.abs((coef_k_soft * X[:, n]).sum())
    tempo = (coef_k * F_coef_BPM).sum()
    corr = np.abs((coef_k * X[:, n]).sum())


    for k in range(F_coef_BPM.shape[0]):
        # kernel, t_kernel, t_kernel_sec = libfmp.c6.compute_sinusoid_optimal(X[k,n], 
        #                                         F_coef_BPM[k], n, Fs_nov, N, H)
        kernel, t_kernel, t_kernel_sec = compute_sinusoid_optimal(X[k,n], 
                                                F_coef_BPM[k], n, Fs_nov, N, H)
        
        if kernel_sum is None:
            # kernel_sum = coef_k_soft[k] * kernel
            kernel_sum = coef_k[k] * kernel
        else:
            # kernel_sum += coef_k_soft[k] * kernel
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

def genNovRef(fs_nov = 100, total_dur = 10, bpm = 120, start_sec = 0.5, expressive_sec = None):
    nov = np.zeros(int(total_dur*fs_nov))
    ### bpm to hz to ibis
    # hz = = bpm/60
    ibi = 60/bpm
    beat_ref = np.arange(start_sec, total_dur, ibi)
    if expressive_sec ==None:
        beat_ref_expressive = np.copy(beat_ref)
    else:
        beat_ref_expressive = expressive_timing(beat_ref, sigma_sec = expressive_sec)
    beat_frame_ids = (beat_ref_expressive*fs_nov).tolist()
    beat_frame_ids = [int(i) for i in beat_frame_ids]
    nov[beat_frame_ids] = 1
    return nov, beat_ref_expressive, beat_frame_ids, beat_ref

def concatNov(nov_list, beat_ref_list, dur_list):
    nov_cat = np.hstack(nov_list)
    beat_ref_list_new = [beat_ref_list[0]]
    dur_acc = 0
    # print(nov_cat.shape)
    for ind, dur in enumerate(dur_list[:-1]):
        # break
        # beat_ref_list_new[ind+1] = beat_ref_list_new[ind+1]+dur
        dur_acc +=dur
        beat_ref_list_new.append(beat_ref_list[ind+1]+dur_acc)
    beat_ref_cat = np.hstack(beat_ref_list_new)
    return nov_cat, beat_ref_cat       

def gaussian(nov, sigma = 0, max_norm = True):
    if sigma==0:
        nov_gs = nov
    else:
        nov_gs = gaussian_filter1d(nov, sigma=sigma, mode = 'constant')
    if max_norm:
        nov_gs = nov_gs/nov_gs.max()
    return nov_gs

def gaussian_noise(nov, mean = 0, std_deviation = 0.15, rectify = True ): 
    ## if add & norm: ground truth may be shifted, so now only replace 
    noise = np.random.normal(loc=mean, scale = std_deviation, size = len(nov))
    if rectify == True:
        noise =np.abs(noise)
    cat = np.hstack([nov[:, np.newaxis], noise[:, np.newaxis]])
    # print(cat.shape)
    nov_noise = cat.max(axis = 1)
    
    return nov_noise

def expressive_timing(beat_ref, sigma_sec = 0.03):
    ### skip the first and last beat incase out of boundary
    expres_ref = np.copy(beat_ref)
    noise = np.random.normal(loc=0, scale = sigma_sec, size = len(beat_ref)-2)
    expres_ref[1:-1] = expres_ref[1:-1] + noise
    return expres_ref


def amplitude_distortion(nov, min_amp=0.2, max_amp=1, gaussian_smooth = True):
    # phase = np.random.rand(nov.shape[0])*2*np.pi
    # sinewav_abs = np.abs(np.sin(phase))
    # sinewav_gs = gaussian(sinewav_abs, sigma =6)
    # nov_distorted = nov* sinewav_gs
    if min_amp==1: 
        nov_distorted = nov
    else:
        ratio = np.clip(np.random.rand(nov.shape[0]), a_min = min_amp, a_max = max_amp)
        # ratio_gs = gaussian(ratio, sigma= 6)
        nov_distorted = ratio*nov
    return nov_distorted

def remove_peaks(nov_binary, ratio = 0.05):
    if ratio==0:
        nov_binary_new = np.copy(nov_binary)
    else:
        pks = np.where(nov_binary==1)[0].tolist()
        num2rm = int(ratio * len(pks))
        random.shuffle(pks)
        rm_ids = pks[:num2rm]
        nov_binary_new = np.copy(nov_binary)
        nov_binary_new[rm_ids] = 0
        # keep_ids = sorted(ids[:-num2rm])
        # nov_binary_new = nov_binary[keep_ids]
    return nov_binary_new

def add_peaks(nov_binary, ratio = 0.2):
    ### different ratio caculation: #beats *ratio = peaks to add
    if ratio==0:
        nov_binary_new = np.copy(nov_binary)
    else:
        nonpks = np.where(nov_binary==0)[0].tolist()
        # num2add = int(ratio * len(nonpks))
        num2add = int(ratio * nov_binary.sum())
        random.shuffle(nonpks)
        add_ids = nonpks[:num2add]
        nov_binary_new = np.copy(nov_binary)
        nov_binary_new[add_ids ] = 1
        # print(nov_binary.sum(), nov_binary_new.sum())
        # keep_ids = sorted(ids[:-num2rm])
        # nov_binary_new = nov_binary[keep_ids]

    return nov_binary_new

def read_anncsv(csvpath):
    """ Read the provided .csv and organize the four types of annotations
    
    Args:
        csvpath (str): Filepath of annotations.
    Returns:
        ann_dict (dict): Dictionary of annotations for onset, beat, downbeat, and structure. 
    """
    ann_info = np.array(pd.read_csv(csvpath, header = None))
    ann_dict = {'onset':[], 'beat':[], 'downbeat':[], 'structure':[]}
    for ann_str in ann_info[1:]:
        time, ann_type = ann_str[0].split(';')
        ann_dict[ann_type.strip('"')].append(float(time))
    return ann_dict

def genNovelty(
    # tempo_list = [120], # BPM, 
    # dur_list = [10], # sec, 
    # start_sec_list = [0.5], #sec, 
    # total_sec = 20, 
    # start_sec = 0.5, 
    # initial_bpm = 30, 
    # tempo_change_ratio = 1.1, 
    example = 'Brahms',
    fs_nov = 100, # FPS
    add_ratio = 0,
    remove_ratio = 0,
    expressive_sec = 0,  ### using original beat_ref by default
    gs_smth_sigma = 3, 
    amp_distort_min = 1.0, 
    amp_distort_max = 1.0, 
    gauss_noise_sigma = 0, 
    plot_start_sec = 0, 
    plot_dur_sec = 100,
    # save = False, 
    figsize = (10, 2),

):
    
    ### generate idea nov
    # nov_gen, beat_ref_cat = genIncreasingTempoNov(total_sec = total_sec, start_sec = start_sec, 
    #                       initial_bpm = initial_bpm, tempo_change_ratio = tempo_change_ratio, 
    #                       fs_nov = fs_nov)
    # # beat_ref_cat = np.copy(nov_gen)
    # nov_cat = np.copy(nov_gen)
    ### prepare nov for the assigned example
    data_dir = '/home/ALABSAD/sunnycyc/GroupMM/AL_Work/WorkCC/projects/DFG-Learn/DPLP/basic-experiments/data/data_fmp/'
    exp_dir = os.path.join(data_dir, 'figure1_novelty_{}/'.format(example))
    csv_path = glob.glob(os.path.join(exp_dir, "*annotations.csv"))[0]
    ann_dict = read_anncsv(csv_path )
    beat_ref_cat = np.array(ann_dict['beat'])
    audio_path = glob.glob(os.path.join(exp_dir, "*.wav"))[0]
    print('audio_path: ', audio_path)
    x, Fs = librosa.load(audio_path, sr = 22050) 
    print('x shape:{}, fs:{}'.format(x.shape, Fs))
    nov_path = os.path.join(exp_dir, 'nov_spectrum_resample1.npy')
    if not os.path.exists(nov_path):
        # nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=441, 
        #                                          gamma=100, M=10, norm=True)
        nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512, 
                                                 gamma=100, M=10, norm=True)
        nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)
    else:
        nov = np.load(nov_path)
    print('nov shape:{}'.format(nov.shape))
    
    ### randomly add peaks
    nov = add_peaks(nov, ratio = add_ratio)
    ### randomly remove peaks from novelty, but the beat annotation stays the same (e.g., like rest notes)
    nov = remove_peaks(nov, ratio = remove_ratio)
    ### apply random amplitude distortion
    nov = amplitude_distortion(nov, min_amp = amp_distort_min, max_amp = amp_distort_max)
    ### apply Gaussian smoothing
    nov = gaussian(nov, sigma = gs_smth_sigma)
    
    ### add Gaussian noise
    nov = gaussian_noise(nov, mean = 0, std_deviation = gauss_noise_sigma)
    
    ### visualize
    start_frame = plot_start_sec * fs_nov
    dur_frame = plot_dur_sec *fs_nov
    end_frame = min(len(nov), start_frame + dur_frame)
    
    
    plt.figure(figsize = figsize)
    plt.plot(nov, label = 'nov', color = 'black')
    plt.vlines(beat_ref_cat*fs_nov, 0, 1, linestyle = 'dashed', color = 'red', label = 'ref')

    plt.xlim([start_frame, end_frame])
    plt.xlabel('Time (frame, FPS={})'.format(fs_nov))
    plt.ylabel('Amplitude')
    plt.legend(bbox_to_anchor = (1.0, 1))
    plt.show()
    # if save:
    save_dir = '/home/ALABSAD/sunnycyc/GroupMM/AL_Work/WorkCC/projects/DFG-Learn/DPLP/basic-experiments/temp_data/'
    pk_path = os.path.join(save_dir, 'novelty.pickle')
    nov_dict = {
        'nov': nov, 
        'beat_ref': beat_ref_cat,
        'audio_path': audio_path, 
        # 'total_sec': total_sec, 
        # 'start_sec': start_sec, 
        # 'initial_bpm': initial_bpm, 
        # 'tempo_change_ratio': tempo_change_ratio, 
        # 'tempo_list': tempo_list, # BPM, 
        # 'dur_list': dur_list, # sec, 
        # 'start_sec_list' : start_sec_list, #sec, 
        'fs_nov': fs_nov, # FPS
        'add_ratio': add_ratio,
        'remove_ratio': remove_ratio,
        'expressive_sec': expressive_sec,  ### using original beat_ref by default
        'gs_smth_sigma': gs_smth_sigma, 
        'amp_distort_min': amp_distort_min, 
        'amp_distort_max': amp_distort_max, 
        'gauss_noise_sigma': gauss_noise_sigma, 
        
    }
    print('saving to :', pk_path)
    with open(pk_path, 'wb') as file:
        pickle.dump(nov_dict, file)

def genNovelty_interactive(
    # tempo_list = [120], # BPM, 
    # dur_list = [10], # sec, 
    # start_sec_list = [0.5], #sec,
    # total_sec = 20, 
    # start_sec = 0.5, 
    # initial_bpm = 30, 
    # tempo_change_ratio = 1.1, 
    example = 'Brahms',
    fs_nov = 100, # FPS
    add_ratio = 0, 
    remove_ratio = 0,
    expressive_sec = 0,  ### using original beat_ref by default
    gs_smth_sigma = 3, 
    amp_distort_min = 1.0, 
    amp_distort_max = 1.0, 
    gauss_noise_sigma = 0, 
    plot_start_sec = 0, 
    plot_dur_sec = 100,
    # save = False,
    ):
    
    return genNovelty(
    # tempo_list = tempo_list , # BPM, 
    # dur_list = dur_list, # sec, 
    # start_sec_list = start_sec_list, #sec, 
    # total_sec = total_sec, 
    # start_sec = start_sec, 
    # initial_bpm = initial_bpm, 
    # tempo_change_ratio = tempo_change_ratio, 
    example = example,
    fs_nov = fs_nov, # FPS
    add_ratio = add_ratio,
    remove_ratio = remove_ratio,
    expressive_sec = expressive_sec,  ### using original beat_ref by default
    gs_smth_sigma = gs_smth_sigma, 
    amp_distort_min = amp_distort_min, 
    amp_distort_max = amp_distort_max, 
    gauss_noise_sigma = gauss_noise_sigma, 
    plot_start_sec = plot_start_sec, 
    plot_dur_sec = plot_dur_sec, 
    # save = save,
    )

def genIBIs(start_frame, initial_bpm = 30, total_sec = 20, fs_nov = 100, 
            tempo_change_ratio = 2, max_tempo = 800):
    ibi_min = 60*fs_nov/max_tempo
    ibi_ori = int(round(60*fs_nov/initial_bpm))
    ibis = [ibi_ori]
    
    id_acc = int(round(ibi_ori))+start_frame
    ids =[start_frame, id_acc]
    run = True
    while run:
        ibi_next = ibi_ori/tempo_change_ratio
        run = (id_acc<total_sec*fs_nov-1)
        if ibi_next>= ibi_min:
            ibi_ori = int(round(ibi_next))
            id_acc += ibi_ori
            run = (id_acc<total_sec*fs_nov-1)
            if run:
                ibis.append(ibi_ori)
                ids.append(id_acc)
        
            
        else:
            run = False
    
    
    return ids, ibis

def genIncreasingTempoNov(total_sec = 20, start_sec = 0.5, 
                          initial_bpm = 30, tempo_change_ratio = 2, 
                          fs_nov = 100, max_tempo = 480):
    """
    Goal: generate a binary novely function with increasing tempo controlled by a 
    tempo_change_ratio

    Parameters
    ----------
    total_sec : TYPE, optional
        DESCRIPTION. The default is 20.
    start_sec : TYPE, optional
        DESCRIPTION. The default is 0.5.
    initial_bpm : TYPE, optional
        DESCRIPTION. The default is 60.
    tempo_change_ratio : TYPE, optional
        DESCRIPTION. The default is 2.
    fs_nov : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------

    """
    nov = np.zeros(int(fs_nov * total_sec))
    start_frame = int(round(start_sec*fs_nov))
    ### get beat ids
    beat_ids, _ = genIBIs(start_frame, initial_bpm = initial_bpm, total_sec = total_sec, 
                          tempo_change_ratio = tempo_change_ratio, 
                          fs_nov = fs_nov, max_tempo = max_tempo)
    beat_ids_array = np.array(beat_ids)
    nov[beat_ids_array] = 1
    
    return nov, beat_ids_array/fs_nov
