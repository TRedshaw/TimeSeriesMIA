import sdv

from sdv.single_table import CTGANSynthesizer

from collections import Counter
import pandas as pd
import numpy as np
import wfdb
import os
from scipy.signal import find_peaks


root = '/Users/toby/PycharmProjects/Year4Project/Year4Project/mia_against_time_series'
# First we are going to check the data for the MIT arrhythmia dataset which is popular and well-annotated
path = os.path.join(root, './mit_raw_data')
records_list = sorted(list(set([f.split('.')[0] for f in os.listdir(path) if '.dat' in f])))

sample = records_list[0]


# To read a record:
record = wfdb.rdrecord( os.path.join( path, sample), sampfrom=100000, sampto=100000+3600)
# wfdb.plot_wfdb(record, title=f'Record {sample} from MIT Arrhythmia dataset')

activations = []
for i in records_list:
    # Get the signal
    signal, field = wfdb.rdsamp(os.path.join( path, i), channels=[0])
    signal = signal[:650000-2000].reshape(650000//3600, 3600, 1)
    activations.append( signal)

activations = np.concatenate(activations, axis = 0)


activations = (activations - np.min(activations, axis = 1)[:,np.newaxis, :])/(np.max(activations, axis = 1)[:,np.newaxis, :] - np.min(activations, axis = 1)[:,np.newaxis, :])

grads = np.gradient(activations, axis = 1)

zero_crossings = np.where( np.diff( np.sign(grads), axis = 1)>0)[0]

ecgs = []

for i in range(len(activations)):
    zero_crossings = np.where(np.diff(np.sign(grads[i].squeeze())))[0]
    local_maxima = activations[i][zero_crossings]
    local_maxima = (local_maxima - np.min(local_maxima)) / (np.max(local_maxima) - np.min(local_maxima))
    local_maxima_locs = zero_crossings[local_maxima.squeeze() > 0.8]
    local_maxima = activations[i][local_maxima_locs.squeeze()]

    median_interval = np.median(np.diff(local_maxima_locs))
    try:
        signal_length = np.arange(int(median_interval * 1.2))
    except:
        print(i)
        break
        continue
    signal_idx = local_maxima_locs[..., np.newaxis] + signal_length[np.newaxis]
    signal_idx = signal_idx[(signal_idx < 3600).all(axis=1)]  # keep only the ones that do not go over the end
    ecgs.append(activations[i][signal_idx])

from scipy.signal import find_peaks

distance = 10 # To avoid the major problem with the above method, we get 2-3 points around every peak, this will mess up the median afterwards
max_peak = np.max(grads[254].squeeze())
peak_times, heights = find_peaks(grads[254].squeeze(), height = max_peak*0.6, distance = distance)

distance = 10
ecgs = []
for i in range(len(activations)):

    max_peak = np.max(grads[i].squeeze())
    peaks, heights = find_peaks(grads[i].squeeze(), height=max_peak * 0.6, distance=distance)
    median_interval = np.median(np.diff(peaks))
    try:
        signal_length = np.arange(int(median_interval * 1.2))
    except:
        print(i)
        continue
    signal_idx = peaks[..., np.newaxis] + signal_length[np.newaxis]
    signal_idx = signal_idx[(signal_idx < 3600).all(axis=1)]  # keep only the ones that do not go over the end
    ecgs.append(activations[i][signal_idx])

activations = []
annotations = []
annotations_df = pd.DataFrame({})
ecgs = []
ecgs_ann = []

for i in records_list:
    patient_id = i
    # Get the signal from the patient
    signal, field = wfdb.rdsamp(os.path.join(path, i), channels=[0])

    # Get the annotation from the patient
    ann = wfdb.rdann(os.path.join(path, i), 'atr')

    # Fill the annotation with -1 everywhere
    annotation = np.ones(signal.shape).squeeze() * (-1)

    # Get the annotation labels
    annotation_labels = ann.symbol

    # The ann.sample is an array like this: array([    18,     77,    370, ..., 649484, 649734, 649991]). At every one of these points we have an annotation
    # The annotation that we have at each point is accessible via annotation_labels = ann.symbol
    # Now, at the indices that are given by ann.sample, we add just the indices of the ann.symbols that are there. So annotation[18] gives us ann.sample[0]
    annotation[ann.sample] = np.arange(len(ann.sample))

    # Reshape these
    signal = signal[:650000 - 2000].reshape(650000 // 3600, 3600, 1)
    annotation = annotation[:650000 - 2000].reshape(650000 // 3600, 3600, 1)

    # Normalizing like in the individual example above
    signal = (signal - np.min(signal, axis=1)[:, np.newaxis, :]) / (
                np.max(signal, axis=1)[:, np.newaxis, :] - np.min(signal, axis=1)[:, np.newaxis, :])
    grads = np.gradient(signal, axis=1)

    ecgs = []
    ecgs_ann = []
    for idx in range(len(signal)):
        max_peak = np.max(grads[idx].squeeze())
        peaks, heights = find_peaks(grads[idx].squeeze(), height=max_peak * 0.6, distance=distance)
        median_interval = np.median(np.diff(peaks))

        try:
            signal_length = np.arange(int(median_interval * 1.2))
        except:
            print(idx)
            continue
        signal_idx = peaks[..., np.newaxis] + signal_length[np.newaxis]
        signal_idx = signal_idx[(signal_idx < 3600).all(axis=1)]  # keep only the ones that do not go over the end
        ecgs.append(signal[idx][signal_idx])
        ecgs_ann.append(annotation[idx][signal_idx])

    max_beat_length = np.max([f.shape[1] for f in ecgs])

    ecgs = [np.pad(ecg_array, pad_width=((0, 0), (0, max_beat_length - ecg_array.shape[1]), (0, 0)), mode='constant',
                   constant_values=0) for ecg_array in ecgs]
    ecgs_ann = [
        np.pad(ecg_array, pad_width=((0, 0), (0, max_beat_length - ecg_array.shape[1]), (0, 0)), mode='constant',
               constant_values=-1) for ecg_array in ecgs_ann]

    ecgs = np.concatenate(ecgs)
    ecgs_ann = np.concatenate(ecgs_ann)

    patient_df = pd.DataFrame(columns=list(map(str, np.arange(ecgs.shape[1]))), index=np.arange(len(ecgs)))
    patient_df['Class'] = 'o'
    patient_df['Patient'] = patient_id

    for beat_idx in range(len(ecgs)):
        patient_df.iloc[beat_idx, :-2] = ecgs[beat_idx].squeeze()
        patient_df.iloc[beat_idx, -2] = ''.join(
            [ann.symbol[int(f)] for f in ecgs_ann[beat_idx][ecgs_ann[beat_idx] > 0]])

    patient_df.to_csv(f'/Users/toby/PycharmProjects/Year4Project/Year4Project/mia_against_time_series/arrhythmia_dataset_new/{patient_id}.csv')