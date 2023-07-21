from fastapi import FastAPI, File
import io
import csv
import os
import librosa
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import scipy
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import my_dataset
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

app = FastAPI(title="SoundRecognition")
@app.post("files/upload-files")
def upload_files(audiofile_1 : bytes = File(), audiofile_2 : bytes = File()):
    y1, sr1 = librosa.load(io.BytesIO(audiofile_1), mono=True, duration=1)
    y2, sr2 = librosa.load(io.BytesIO(audiofile_2), mono=True, duration=1)
    #extracting of features
    audio_1_features = my_dataset.get_features(y1, sr1)
    audio_2_features = my_dataset.get_features(y2, sr2)
    audio_1_features.extend(audio_2_features)
    audio_features = np.asarray([audio_1_features])
    #reading dataset to dataframe
    data = pd.read_csv('data/dataset_1.csv', encoding='cp1251')
    #extracting feature table (Х), and labels table (у)
    X = data.iloc[:, 2:-1]
    print(type(X))
    X = preprocessing.StandardScaler().fit(X).transform(X)
    print(type(X))
    y = data["label"].values
    # create and fit model
    model = neighbors.KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    # get prediction of the model
    y_pred = model.predict([])
    # output result
    #print("KNeighborsClassifier")
    print("Train set Accuracy: ", metrics.accuracy_score(y, model.predict(X)))
    similarity = round(100 - mean_absolute_percentage_error(audio_1_features, audio_2_features))
    label = False
    if y_pred[0] == 1:
        label = True
    return {'Access': label, 'Similarity': similarity}



"""
model = LogisticRegression()
model.fit(X_train, y_train)

model = svm.SVC(kernel='rbf', verbose=1)
model.fit(X_train, y_train)
"""