import helpermethods
import numpy as np
import sys
import edgeml_pytorch.utils as utils
from edgeml_pytorch.graph.bonsai import Bonsai
import torch
import time
import pandas as pd
from ecgdetectors import Detectors
from scipy.signal import butter, lfilter,filtfilt, iirnotch
from scipy.signal import freqs
import matplotlib.pyplot as plt
from scipy.signal import medfilt

fs = 250
n = 0.5*fs
f_high = 0.5
cut_off = f_high/n

order = 4

def loadModel(currDir):
    '''
    Load the Saved model and load it to the model using constructor
    Returns two dict one for params and other for hyperParams
    '''
    paramDir = currDir + '/'
    paramDict = {}
    paramDict['W'] = np.load(paramDir + "W.npy")
    paramDict['V'] = np.load(paramDir + "V.npy")
    paramDict['T'] = np.load(paramDir + "T.npy")
    paramDict['Z'] = np.load(paramDir + "Z.npy")
    hyperParamDict = np.load(paramDir + "hyperParam.npy", allow_pickle=True).item()
    return paramDict, hyperParamDict


def pipelinedRpeakExtraction(x, fs):
    
    x = detectors.swt_detector(x)
#     x = detectors.hamilton_detector(x)
#     x = detectors.pan_tompkins_detector(x)
    return x

# def get_mean_nni(nn_intervals, fs):
#     diff_nni = np.diff(nn_intervals)
#     length_int = len(nn_intervals)
#     return np.mean(nn_intervals)

def _2017_top_4_features(nn_intervals, fs):
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)
    
    nni_50 = sum(np.abs(diff_nni) > (50*fs/1000))
    pnni_50 = 100 * nni_50 / length_int
    
    nni_20 = sum(np.abs(diff_nni) > (20*fs/1000))
    cvsd = np.sqrt(np.mean(diff_nni ** 2)) / np.mean(nn_intervals)
    
    return nni_50, pnni_50, nni_20, cvsd

def _2017_top_6_features(nn_intervals, fs):
    
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)
    
    nni_50 = sum(np.abs(diff_nni) > (50*fs/1000))
    pnni_50 = 100 * nni_50 / length_int
    
    nni_20 = sum(np.abs(diff_nni) > (20*fs/1000))
    cvsd = np.sqrt(np.mean(diff_nni ** 2)) / np.mean(nn_intervals)
    
    sdnn = np.std(nn_intervals, ddof = 1)
    cvnni = sdnn / np.mean(nn_intervals)
    
    heart_rate_list = np.divide(60000, nn_intervals)
    max_hr = max(heart_rate_list)
    
    return nni_50, pnni_50, nni_20, cvsd, cvnni, max_hr

def _2017_top_8_features(nn_intervals, fs):
    
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)
    
    nni_50 = sum(np.abs(diff_nni) > (50*fs/1000))
    pnni_50 = 100 * nni_50 / length_int
    
    nni_20 = sum(np.abs(diff_nni) > (20*fs/1000))
    cvsd = np.sqrt(np.mean(diff_nni ** 2)) / np.mean(nn_intervals)
    
    sdnn = np.std(nn_intervals, ddof = 1)
    cvnni = sdnn / np.mean(nn_intervals)
    
    heart_rate_list = np.divide(60000, nn_intervals)
    max_hr = max(heart_rate_list)
    mean_hr = np.mean(heart_rate_list)
    sdnn = np.std(nn_intervals, ddof = 1)
    
    return nni_50, pnni_50, nni_20, cvsd, cvnni, max_hr, mean_hr, sdsd

def _2017_top_10_features(nn_intervals, fs):
    
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)
    
    nni_50 = sum(np.abs(diff_nni) > (50*fs/1000))
    pnni_50 = 100 * nni_50 / length_int
    
    nni_20 = sum(np.abs(diff_nni) > (20*fs/1000))
    cvsd = np.sqrt(np.mean(diff_nni ** 2)) / np.mean(nn_intervals)
    
    sdnn = np.std(nn_intervals, ddof = 1)
    cvnni = sdnn / np.mean(nn_intervals)
    
    heart_rate_list = np.divide(60000, nn_intervals)
    max_hr = max(heart_rate_list)
    mean_hr = np.mean(heart_rate_list)
    sdnn = np.std(nn_intervals, ddof = 1)
    
    std_hr = np.std(heart_rate_list)
    pnni_20 = 100 * nni_20 / length_int
    
    
    return nni_50, pnni_50, nni_20, cvsd, cvnni, max_hr, mean_hr, sdsd, std_hr, pnni_20

def _2017_top_12_features(nn_intervals, fs):
    
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)
    
    nni_50 = sum(np.abs(diff_nni) > (50*fs/1000))
    pnni_50 = 100 * nni_50 / length_int
    
    nni_20 = sum(np.abs(diff_nni) > (20*fs/1000))
    cvsd = np.sqrt(np.mean(diff_nni ** 2)) / np.mean(nn_intervals)
    
    sdnn = np.std(nn_intervals, ddof = 1)
    cvnni = sdnn / np.mean(nn_intervals)
    
    heart_rate_list = np.divide(60000, nn_intervals)
    max_hr = max(heart_rate_list)
    mean_hr = np.mean(heart_rate_list)
    sdnn = np.std(nn_intervals, ddof = 1)
    
    std_hr = np.std(heart_rate_list)
    pnni_20 = 100 * nni_20 / length_int
    
    rmssd = np.sqrt(np.mean(diff_nni ** 2))
    sdnn = np.std(nn_intervals, ddof = 1)
    
    return nni_50, pnni_50, nni_20, cvsd, cvnni, max_hr, mean_hr, sdsd, std_hr, pnni_20, rmssd, sdnn


    

def afdb_top_4_features(nn_intervals, fs): 
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)
    
    nni_50 = sum(np.abs(diff_nni) > (50*fs/1000))
    pnni_50 = 100 * nni_50 / length_int
    
    nni_20 = sum(np.abs(diff_nni) > (20*fs/1000))
    pnni_20 = 100 * nni_20 / length_int
    
    return np.array([nni_20, nni_50, pnni_20, pnni_50, 1])
    
    
def afdb_top_6_features(nn_intervals, fs):
    
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)
    
    nni_50 = sum(np.abs(diff_nni) > (50*fs/1000))
    pnni_50 = 100 * nni_50 / length_int
    
    nni_20 = sum(np.abs(diff_nni) > (20*fs/1000))
    pnni_20 = 100 * nni_20 / length_int
    
    sdnn = np.std(nn_intervals, ddof = 1)
    cvnni = sdnn / np.mean(nn_intervals)
    cvsd = np.sqrt(np.mean(diff_nni ** 2)) / np.mean(nn_intervals)
    
    return nni_20, nni_50, pnni_20, pnni_50, cvnni, cvsd

def afdb_top_8_features(nn_intervals, fs):
    
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)
    
    nni_50 = sum(np.abs(diff_nni) > (50*fs/1000))
    pnni_50 = 100 * nni_50 / length_int
    
    nni_20 = sum(np.abs(diff_nni) > (20*fs/1000))
    pnni_20 = 100 * nni_20 / length_int
    
    sdnn = np.std(nn_intervals, ddof = 1)
    cvnni = sdnn / np.mean(nn_intervals)
    cvsd = np.sqrt(np.mean(diff_nni ** 2)) / np.mean(nn_intervals)
    
    heart_rate_list = np.divide(60000, nn_intervals)
    std_hr = np.std(heart_rate_list)
    return nni_20, nni_50, pnni_20, pnni_50, cvnni, cvsd, sdnn, std_hr 

def afdb_top_10_features(nn_intervals, fs):
    
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)
    
    nni_50 = sum(np.abs(diff_nni) > (50*fs/1000))
    pnni_50 = 100 * nni_50 / length_int
    
    nni_20 = sum(np.abs(diff_nni) > (20*fs/1000))
    pnni_20 = 100 * nni_20 / length_int
    
    sdnn = np.std(nn_intervals, ddof = 1)
    cvnni = sdnn / np.mean(nn_intervals)
    cvsd = np.sqrt(np.mean(diff_nni ** 2)) / np.mean(nn_intervals)
    
    heart_rate_list = np.divide(60000, nn_intervals)
    std_hr = np.std(heart_rate_list)
    
    max_hr = max(heart_rate_list)
    sdsd = np.std(diff_nni)
    
    return nni_20, nni_50, pnni_20, pnni_50, cvnni, cvsd, sdnn, std_hr, max_hr, sdsd

def afdb_top_12_features(nn_intervals, fs):
    
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)
    
    nni_50 = sum(np.abs(diff_nni) > (50*fs/1000))
    pnni_50 = 100 * nni_50 / length_int
    
    nni_20 = sum(np.abs(diff_nni) > (20*fs/1000))
    pnni_20 = 100 * nni_20 / length_int
    
    sdnn = np.std(nn_intervals, ddof = 1)
    cvnni = sdnn / np.mean(nn_intervals)
    cvsd = np.sqrt(np.mean(diff_nni ** 2)) / np.mean(nn_intervals)
    
    heart_rate_list = np.divide(60000, nn_intervals)
    std_hr = np.std(heart_rate_list)
    
    max_hr = max(heart_rate_list)
    sdsd = np.std(diff_nni)
    
    mean_hr = np.mean(heart_rate_list)
    rmssd = np.sqrt(np.mean(diff_nni ** 2))
    
    return nni_20, nni_50, pnni_20, pnni_50, cvnni, cvsd, sdnn, std_hr, max_hr, sdsd, mean_hr, rmssd

def afdb_top_14_features(nn_intervals, fs):
    
    diff_nni = np.diff(nn_intervals)
    abs_diff = np.abs(diff_nni)
    length_int = len(nn_intervals)
    mean_nni = np.mean(nn_intervals)
    rmssd = np.sqrt(np.mean(diff_nni ** 2))
    
    
    nni_50 = sum(abs_diff > 12.5)
    pnni_50 = 100 * nni_50 / length_int
    
    nni_20 = sum(abs_diff > 5)
    pnni_20 = 100 * nni_20 / length_int
    
    sdnn = np.std(nn_intervals, ddof = 1)
    cvnni = sdnn / mean_nni
    cvsd = rmssd / mean_nni
    
    heart_rate_list = np.divide(60, nn_intervals)
    std_hr = np.std(heart_rate_list)
    
    max_hr = max(heart_rate_list)
    sdsd = np.std(diff_nni)
    
    mean_hr = np.mean(heart_rate_list)
    
    min_hr = min(heart_rate_list)
    
    
    return np.array([nni_20, nni_50, pnni_20, pnni_50, cvnni, cvsd, sdnn, std_hr, max_hr, sdsd, mean_hr, rmssd, min_hr, mean_nni, 1])

def _2017_top_14_features(nn_intervals, fs):
    
    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)
    
    nni_50 = sum(np.abs(diff_nni) > (50*fs/1000))
    pnni_50 = 100 * nni_50 / length_int
    
    nni_20 = sum(np.abs(diff_nni) > (20*fs/1000))
    cvsd = np.sqrt(np.mean(diff_nni ** 2)) / np.mean(nn_intervals)
    
    sdnn = np.std(nn_intervals, ddof = 1)
    cvnni = sdnn / np.mean(nn_intervals)
    
    heart_rate_list = np.divide(60000, nn_intervals)
    max_hr = max(heart_rate_list)
    mean_hr = np.mean(heart_rate_list)
    sdnn = np.std(nn_intervals, ddof = 1)
    
    std_hr = np.std(heart_rate_list)
    pnni_20 = 100 * nni_20 / length_int
    
    rmssd = np.sqrt(np.mean(diff_nni ** 2))
    sdnn = np.std(nn_intervals, ddof = 1)
    
    min_hr = min(heart_rate_list)
    mean_nni = np.mean(nn_intervals)
    
    return nni_50, pnni_50, nni_20, cvsd, cvnni, max_hr, mean_hr, sdsd, std_hr, pnni_20, rmssd, sdnn, min_hr, mean_nni


device = torch.device("cpu")

MODEL_DIR = "/hdd/physio/edgeml/examples/pytorch/Bonsai/AFDB_top14/PyTorchBonsaiResults/16_50_10_09_08_21"

paramDict, hyperParamDict = loadModel(MODEL_DIR)

bonsai = Bonsai(hyperParamDict['numClasses'], hyperParamDict['dataDim'], hyperParamDict['projDim'],
             hyperParamDict['depth'], hyperParamDict['sigma'], W=paramDict['W'], T=paramDict['T'], V=paramDict['V'],
            Z=paramDict['Z']).to(device)

sigmaI = 1e9

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


window = np.load("1window.npy")

fs = 250
detectors = Detectors(fs)
b, a = butter(order, cut_off,btype='high')

times = []
for i in range(int(sys.argv[1])):
    start = time.time() 
    window = filtfilt(b, a, window)
    window = normalize(window)
    x = pipelinedRpeakExtraction(window, fs)
    x = np.diff(x)
    features = afdb_top_14_features(x, fs)
    _, _ = bonsai(torch.from_numpy(features.astype(np.float32)), sigmaI)
    end = time.time()
    times.append(end - start)
    

print("features + model + baseline wander removal: ", np.mean(times)*1000, "ms")

times = []
for i in range(int(sys.argv[1])):
    start = time.time() 
    x = pipelinedRpeakExtraction(window, fs)
    x = np.diff(x)
    features = afdb_top_14_features(x, fs)
    _, _ = bonsai(torch.from_numpy(features.astype(np.float32)), sigmaI)
    end = time.time()
    times.append(end - start)
    

print("features + model : ", np.mean(times)*1000, "ms")
print("features + model : max", np.max(times)*1000, "ms")
print("features + model : min", np.min(times)*1000, "ms")

times = []
for i in range(int(sys.argv[1])): 
    start = time.time()
    _, _ = bonsai(torch.from_numpy(features.astype(np.float32)), sigmaI)
    end = time.time()
    times.append(end - start)
    

print("model : ", np.mean(times)*1000, "ms")