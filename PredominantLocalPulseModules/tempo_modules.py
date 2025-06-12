# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 14:09:38 2023


@author: sunnycyc
"""

#%%
import os
import numpy as np
import math


def getIBIcurve(beat_file, ):
    beats = np.loadtxt(beat_file)
    
    if len(beats.shape)<2:
        beats = beats[:, np.newaxis]
    ### if the annotation contains less than 2 beats:
    if len(beats)<2:
        print("===> Warning: Less than 2 beats within:{}".format(beat_file))
        print("===> return 0 IBI")
        return 0
    else:
        ibis = beats[1:, 0]-beats[0:-1, 0]
    return ibis

def stableRatio(curve, deviation = 0.04):
    
    mean = curve.mean()
    curve_norm = curve/mean
    within_range = (curve_norm >= 1-deviation) & (curve_norm <= 1 + deviation)
    stab_ratio = within_range.mean()
    
    return stab_ratio

# stab_ibi = stableRatio(ibis)
# stab_tempi = stableRatio(tempocurve)
# print(stab_ibi, stab_tempi)

def tempoStability(beat_file, deviation = 0.04):
    tempocurve = getTempoCurve(beat_file)
    tempo_stability = stableRatio(tempocurve, deviation = deviation)
    return tempo_stability
    
    

def getTempoCurve(beat_file, ):
    """
    read beat annotations, and use the ibi curve to derive the tempo curve

    Parameters
    ----------
    beat_file : annotation path in string format

    Returns
    -------
    tempocurve : np.array
        a sequence of tempi, converted from ibi curve, 

    """
    ibis = getIBIcurve(beat_file)

    tempocurve = 60/ibis
    
    return tempocurve
   

def getSongMeanTempo(beat_file, mean_type = 'ori_ibi', smooth_winlen = None):
    """
    Calculate mean tempo based on mean IBI of reference beats in input beat_file
    
    Parameters
    ----------
    beat_file : str
        path of beat annotation for a specific song.
    mean_type : str, optional
        method name for calculating mean_tempo of the song. The default is 'ori_ibi'.
    smooth_winlen : int, optional
        if requires smoothing when using different mean_type, could be use to decide
        window length. The default is None.

    Returns
    -------
    mean_tempo: float
        Mean tempo of the input beat_file in BPM.


    """
   
    ibis = getIBIcurve(beat_file)
    ### mean tempo of each song can be calculated in different ways
    ### "ori_ibi": using raw inter-beat-intervals without any smoothing
    if mean_type =='ori_ibi':
        mean_ibi = ibis.mean() # in sec
        mean_tempo = 60/mean_ibi
    else:
        print("Haven't implement this mean_type:", mean_type)
        mean_tempo =  None

    return mean_tempo



def calCvar(tempocurve):
    """
    input:
        tempocurve: np.array with shape (#beat intervals, )

    output: 
        Coefficient of variation = std/mean """
    return tempocurve.std()/tempocurve.mean()

def calLocalCvar(tempocurve, padding = True, hop = 1, win_len = 12):
    """ 
    input:
        tempocurve: 12sec median tempo curve, np array shape (time, )
        padding: zeropadding to ensure output len == tempocurve
        hop: hopsize, default =  1 sec
        win_len: window length, default = 12 secs
        
    output:
        localCvar curve
    """
    # number of 12 second windows
    if padding:
        tempocurve = np.hstack([tempocurve, np.zeros((win_len-1,))+tempocurve[-1]])
    
    num_win = math.ceil((len(tempocurve)-win_len)/(hop)+1)
    
    localCvar = []
    for w_ind in range(num_win):
        # break
        start_frame = w_ind*hop
        end_frame = min((start_frame + win_len), len(tempocurve))
        # print("start:{}, end:{}".format(start_frame, end_frame))
        check_region = tempocurve[start_frame:end_frame]
        
        localCvar.append(calCvar(check_region))
    return localCvar

#### functions for L-correct
def countSegCorrect(reference, beat_est, tolerance = 0.07):
    ## find time range of the L correct beats for this segment 
    start = reference[0] - tolerance
    end = reference[-1] + tolerance
    detectedBeats_ind = (start<=beat_est)& (beat_est<=end)
    detectedBeats = beat_est[detectedBeats_ind]
    
    tmp = 0
    if len(reference)!= len(detectedBeats):
        ### not L-correct, directly pass
        return False
    else:
        tmp = 1 ### assume this segment is L-correct, and conduct further check
        for ref_ind in range(len(reference)):
            # break
            toll = reference[ref_ind]- tolerance # lower bound
            tolh = reference[ref_ind]+ tolerance # higher bound
            ## if number of detectedBeats in between lower/higher bounds !=1, not correct
            est_within_id = (detectedBeats>=toll) &(detectedBeats<=tolh)
            if sum(est_within_id)!=1:
                tmp = 0
    if tmp ==1:
        return True
    else:
        return False
    
def calScore(segCorrect, beat_est, beat_ref, eval_type = 'onbeat', eps = 1e-16):
    """ input:
            segCorrect (np.array, dtype = bool): indicating the correct beats, 
            beat_ref (np.array, dtype = float): indicating the annotated beat times, 
            eval_type (str): could be 'onbeat', 'half', 'third', 'any'
        output:
            result (dict): keys including 'eval_type', 'f-score', 'precision', 'recall'
    """
    Recall = sum(segCorrect)/len(beat_ref)
    Precision = sum(segCorrect)/len(beat_est)
    Fscore = (2*Recall*Precision)/(Recall + Precision + eps)
    result = {
            'F-'+eval_type: Fscore, 
            'P-'+eval_type: Precision, 
            'R-'+eval_type: Recall, 
        }
    # result = {
    #     eval_type: {
    #                 'Fscore': Fscore, 
    #                 'Precision': Precision, 
    #                 'Recall': Recall, 
    #                 },
    #     }
    return result

def Lcorrect_eval(beat_est, beat_ref, tolerance = 0.07, L = 4):
    """ 
    input:
        beat_est: array of estimated beat positions (sec), 
        beat_ref: array of reference beat positions (sec), 
        tolerance: tolerance for evaluation (sec), 
        L: number of continuous frames required to be correct (int)
    
    output:
        normal_F: f-score for normal cases, 
        half_F: f-score when treating half-beats as correct, 
        third_f: f-score when treating third-beats as correct, 
        any_f: f-score when treating all above cases as correct, 
    """
    
    ### prepare index-start and index-end for each frame's evaluation
    segStarts = np.arange(0, len(beat_ref)).tolist()
    segEnds = np.arange(L, len(beat_ref)+1).tolist()
    ## for final l-1 frames, use final beat
    if len(segEnds)< len(segStarts):
        segEnds +=[segStarts[-1]+1]*(len(segStarts)-len(segEnds)) 
    
    ### creat arrays to save correct beats
    segCorrect = np.zeros(len(segStarts))
    segHalfCorrect = np.zeros(len(segStarts))
    segOneThirdCorrect = np.zeros(len(segStarts))
    segTwoThirdCorrect = np.zeros(len(segStarts))
    
    
    ### for each step/segment(frame or beat), extract correct reference beats and check if L-correct
    for segment in segStarts:
        # break
        # if segment ==51:
        #     break
        # print("start:{}, end:{}".format(segStarts[segment], segEnds[segment]))
        reference = beat_ref[segStarts[segment]:segEnds[segment]]
        # print("ref:{}".format(reference))
        
        # print("start:", segStarts[segment:])
        # print("end:", segEnds[segment:])
        
        
        #### counpound beat reference
        if segEnds[segment] < len(beat_ref):
            ref_next = beat_ref[segStarts[segment]+1: segEnds[segment]+1]
            
            # print("next:", ref_next)
            # print("======")
            ref_half_beat = reference + 0.5 *(ref_next-reference)
            ref_oneThird_beat = reference + (1/3)*(ref_next-reference)
            ref_twoThird_beat = reference + (2/3)*(ref_next-reference)
        else:
            ref_last = beat_ref[segStarts[segment]-1: segEnds[segment]-1]
            ref_half_beat = reference + 0.5*(reference - ref_last)
            ref_oneThird_beat = reference +(1/3)*(reference - ref_last)
            ref_twoThird_beat = reference +(2/3)*(reference - ref_last)

        #### counting number of correct beats
        ### count correct for on beat
        if countSegCorrect(reference, beat_est, tolerance = tolerance):
            segCorrect[segStarts[segment]:segEnds[segment]]=1
        ### count correct for half beat
        if countSegCorrect(ref_half_beat, beat_est, tolerance = tolerance):
            segHalfCorrect[segStarts[segment]:segEnds[segment]] = 1
        ### couint correct for oneThird beat
        if countSegCorrect(ref_oneThird_beat, beat_est, tolerance = tolerance):
            segOneThirdCorrect[segStarts[segment]:segEnds[segment]] = 1
        ### couint correct for twoThird beat
        if countSegCorrect(ref_twoThird_beat, beat_est, tolerance = tolerance):
            segTwoThirdCorrect[segStarts[segment]:segEnds[segment]] = 1
            
    ### calculate scores
    segCorrect = np.array(segCorrect, dtype = bool)
    segHalfCorrect = np.array(segHalfCorrect, dtype = bool)
    segOneThirdCorrect = np.array(segOneThirdCorrect, dtype = bool)
    segTwoThirdCorrect = np.array(segTwoThirdCorrect, dtype = bool)
    
    
    segHalfCorrect = segHalfCorrect | segCorrect
    segThirdCorrect = segOneThirdCorrect | segTwoThirdCorrect | segCorrect
    segAnyCorrect = segOneThirdCorrect | segTwoThirdCorrect | segHalfCorrect | segCorrect
    
    all_results = {}
    for segC, evaltype in [(segCorrect, 'onbeat'), 
                           (segHalfCorrect, 'half'), 
                           (segThirdCorrect, 'third'), 
                           (segAnyCorrect, 'any')]:
        all_results.update(calScore(segC, beat_est, beat_ref, eval_type = evaltype))
    return all_results
    