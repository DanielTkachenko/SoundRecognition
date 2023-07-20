import csv
import os
import librosa
import numpy as np

import pandas as pd
import scipy

N_MFCC = 20

def find_match(file1, file2):
    for w in ["Вход", "Пицца", "Привет", "Собака", "Шкаф"]:
        if w in file1 and w in file2:
            return 1
    return 0

def get_features(filepath):
    y, sr = librosa.load(filepath, mono=True)
    features = []
    features.extend([np.mean(e) for e in librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)])  #mfcc_mean
    features.extend([np.std(e) for e in librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)])  # mfcc_std
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T,axis=0)[0])  # cent_mean
    features.append(np.std(librosa.feature.spectral_centroid(y=y, sr=sr).T,axis=0)[0])  # cent_std
    features.append(scipy.stats.skew(librosa.feature.spectral_centroid(y=y, sr=sr).T,axis=0)[0])  # cent_skew
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T,axis=0)[0])  # rolloff_mean
    features.append(np.std(librosa.feature.spectral_rolloff(y=y, sr=sr).T,axis=0)[0])   # roloff_std
    return features

def get_headers(files_n = 2):
    header = []
    for n in range(files_n):
        header.append(f'fname{n}')
    for n in range(files_n):
        header.extend([f'f{n}_mfcc_mean{i}' for i in range(1, 21)])
        header.extend([f'f{n}_mfcc_std{i}' for i in range(1, 21)])
        header.extend([f'f{n}_cent_mean', f'f{n}_cent_std', f'f{n}_cent_skew', f'f{n}_rolloff_mean', f'f{n}_rolloff_std'])
    header.append("label")
    return header

#execute once to create dataset
def create_dataset():
    with open('data/dataset.csv', 'w', encoding='cp1251', newline ='') as file:
        writer = csv.writer(file, delimiter=',')
        h = get_headers()
        writer.writerow(h)
        for directory1, _, filenames1 in os.walk("./Sounds-dataset/"):
            for filename1 in filenames1:
                print(filename1)
                for directory2, _, filenames2 in os.walk("./Sounds-dataset/"):
                    for filename2 in filenames2:
                        if directory1 == directory2 and filename1 == filename2:
                            continue
                        row = [filename1, filename2]
                        row.extend(get_features(os.path.join(directory1, filename1)))
                        row.extend(get_features(os.path.join(directory2, filename2)))
                        if directory1 != directory2:
                            label = 0
                        elif directory1 == directory2:
                            label = find_match(filename1, filename2)
                        row.append(label)
                        writer.writerow(row)