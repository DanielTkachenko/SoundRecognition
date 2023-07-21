import csv
import os
import librosa
import numpy as np

import pandas as pd
import scipy

N_MFCC = 20

filepath = "./data/Sounds-dataset/Алексей/Вход-не распознано Алексей.wav"

y, sr = librosa.load(filepath, mono=True)

def find_match(file1, file2):
    for w in ["Вход", "Пицца", "Привет", "Собака", "Шкаф"]:
        if w in file1 and w in file2:
            return 1
    return 0


def get_features(y, sr):
    features = []
    features.extend([np.mean(e) for e in librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)])  #mfcc_mean
    features.extend([np.std(e) for e in librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)])  # mfcc_std
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T,axis=0)[0])  # cent_mean
    features.append(np.std(librosa.feature.spectral_centroid(y=y, sr=sr).T,axis=0)[0])  # cent_std
    features.append(scipy.stats.skew(librosa.feature.spectral_centroid(y=y, sr=sr).T,axis=0)[0])  # cent_skew
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T,axis=0)[0])  # rolloff_mean
    features.append(np.std(librosa.feature.spectral_rolloff(y=y, sr=sr).T,axis=0)[0])   # roloff_std
    return features

def get_headers(files_n = 2,mfcc_n=20, include_dirs = False):
    header = []
    if include_dirs:
        for n in range(files_n):
            header.append(f'dir{n}')
    for n in range(files_n):
        header.append(f'fname{n}')
    for n in range(files_n):
        header.extend([f'f{n}_mfcc_mean{i}' for i in range(1, mfcc_n+1)])
        header.extend([f'f{n}_mfcc_std{i}' for i in range(1, mfcc_n+1)])
        header.extend([f'f{n}_cent_mean', f'f{n}_cent_std', f'f{n}_cent_skew', f'f{n}_rolloff_mean', f'f{n}_rolloff_std'])
    return header

#execute once to create feature table
def create_features_table(table_file_name):
    with open(table_file_name, 'w', encoding='cp1251', newline ='') as file:
        writer = csv.writer(file, delimiter=',')
        h = get_headers(1, include_dirs=True)
        writer.writerow(h)
        for directory, _, filenames in os.walk("./data/Sounds-dataset/"):
            for filename in filenames:
                row = [directory]
                row.append(filename)
                y, sr = librosa.load(os.path.join(directory, filename), mono=True)
                row.extend(get_features(y, sr))
                writer.writerow(row)


#execute once to create dataset
def feature_table_to_dataset(dataset_file_name, table_file_name):
    with open(dataset_file_name, 'w', encoding='cp1251', newline ='') as file:
        writer = csv.writer(file, delimiter=',')
        h = get_headers()
        h.append("label")
        writer.writerow(h)
        features = pd.read_csv(table_file_name, encoding='cp1251')
        for f1 in features.values:
            for f2 in features.values:
                dir1, dir2 = f1[0], f2[0]
                fname1, fname2 = f1[1], f2[1]
                if dir1 != dir2:
                    continue
                if dir1 == dir2 and fname1 == fname2:
                    continue
                row = []
                row.extend([fname1, fname2])
                row.extend(f1[2:])
                row.extend(f2[2:])
                if dir1 != dir2:
                    label = 0
                elif dir1 == dir2:
                    label = find_match(fname1, fname2)
                row.append(label)
                writer.writerow(row)

