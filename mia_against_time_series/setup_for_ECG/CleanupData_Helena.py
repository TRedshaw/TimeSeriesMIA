import os

import numpy as np
import pandas as pd

classes_dictionary = {'N': 'N',
                      'L': 'N',
                      'R': 'N',
                      'e': 'N',
                      'j': 'N',
                      'A': 'S',
                      'a': 'S',
                      'J': 'S',
                      'S': 'S',
                      'V': 'V',
                      'E': 'V',
                      'F': 'F',
                      '/': 'Q',
                      'f': 'Q',
                      'Q': 'Q',
                      }


def keep_first_classes(row):
    original_class = list(row.Class)

    new_class = ''.join(original_class[:2])
    return new_class


def keep_second_of_two(row):
    first_two_classes = list(row.Class)

    new_class = first_two_classes[-1]
    return new_class


def clean_other_classes(df, classes_dictionary):
    df = df[df['Class'].isin(set(list(classes_dictionary.keys())))]
    return df


def change_class(row, classes_dictionary):
    original_class = list(row.Class)

    new_class = ''.join([classes_dictionary[f] for f in original_class])
    return new_class

record_path = '../arrhythmia_dataset'
median_ecg_length = []
for f in os.listdir(record_path):
    df = pd.read_csv(os.path.join(record_path,f)).drop('Unnamed: 0', axis = 1)
    df = df.dropna()
    df['Class'] = df.apply(lambda x: keep_first_classes(x), axis = 1)
    df['Class']  = df.apply(lambda x: keep_second_of_two(x), axis = 1)
    df = clean_other_classes(df, classes_dictionary)
    df['Class'] = df.apply(lambda x: change_class(x, classes_dictionary), axis = 1)
    l=np.argmin(df.iloc[:,:-2].median())
    print(f'{f} df has shape {df.shape}. With {len(df)} samples and every ecg is {df.shape[1]-2} timepoints long, zeros start at {l}')
    median_ecg_length.append(l)

universal_ecg_length = int(np.max(median_ecg_length))
record_path = '../arrhythmia_dataset'
records = []
for f in os.listdir(record_path):
    df = pd.read_csv(os.path.join(record_path, f)).drop('Unnamed: 0', axis=1)
    df = df.dropna()
    df['Class'] = df.apply(lambda x: keep_first_classes(x), axis=1)
    df['Class'] = df.apply(lambda x: keep_second_of_two(x), axis=1)
    df = clean_other_classes(df, classes_dictionary)
    df['Class'] = df.apply(lambda x: change_class(x, classes_dictionary), axis=1)

    # Cut or pad
    if df.shape[1] >= universal_ecg_length:
        df = df.drop(list(map(str, np.arange(universal_ecg_length, df.shape[1] - 2))), axis=1)
    else:
        df[list(map(str, np.arange(df.shape[1] - 3, universal_ecg_length)))] = 0

    print(f'{f} df has shape {df.shape}. With {len(df)} samples and every ecg is {df.shape[1] - 2} timepoints long')
    records.append(df)

# Records is currently each person's table of ECG split into heartbeats per row, e.g. records[1] is patient 223
print(records)

# This line makes one huge table of all patient tables
records = pd.concat(records)

# print(records)
records.to_csv('/Users/toby/PycharmProjects/Year4Project/Year4Project/mia_against_time_series/arrhythmia_dataset_new/all_records.csv')