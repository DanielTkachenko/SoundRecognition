from fastapi import FastAPI, File
import io
import csv
import os
import librosa
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import scipy

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


import my_dataset
import my_model

#creating dataset (need to execute once)
#my_dataset.create_features_table("./data/features.csv")
#my_dataset.feature_table_to_dataset("./data/dataset_1.csv", "./data/features.csv")

N_MFCC = 20

import wandb
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="SoundRecognition",
    # Track hyperparameters and run metadata
    config={
        "metric": "accuracy",

    })

#getting and fitting of model
model = my_model.create_and_fit_model()

app = FastAPI(title="SoundRecognition")
@app.post("files/upload-files")
def upload_files(audio_1_features, audio_2_features):
    similarity = round(100 - mean_absolute_percentage_error(audio_1_features, audio_2_features))
    audio_1_features.extend(audio_2_features)
    audio_features = np.asarray([audio_1_features])
    # get prediction of the model
    y_pred = model.predict([audio_features])
    # output result
    if y_pred[0] == 1:
        label = True
    else:
        label = False
    return {'Access': label, 'Similarity': similarity}


"""
y1, sr1 = librosa.load(io.BytesIO(audiofile_1), mono=True, duration=1)
y2, sr2 = librosa.load(io.BytesIO(audiofile_2), mono=True, duration=1)
#extracting of features
audio_1_features = my_dataset.get_features(y1, sr1)
audio_2_features = my_dataset.get_features(y2, sr2)
"""