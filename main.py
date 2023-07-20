import csv
import os
import librosa
import numpy as np

import pandas as pd
import scipy

N_MFCC = 20

data = pd.read_csv('data/dataset.csv', encoding='cp1251')
batch = data[:5]
print(batch.columns)
print(batch)