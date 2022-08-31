import argparse

import pandas as pd
import numpy as np
import google.cloud.storage as storage
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, required=True, help='path of dataset input')
parser.add_argument(
    '--bucket_name', type=str, required=True, help='GCS Bucket')

args = parser.parse_args()
dataset=args.dataset
bucket_name = args.bucket_name

param_grid = {'max_depth':[10,15,20],
               'min_samples_split':[10,15,20]}
base_model = RandomForestClassifier(n_estimators = 20, random_state = 2022)

df = pd.read_csv(dataset + '.csv').values
X = df[:,:-1]
y = df[:,-1].astype(int)

oversample = RandomOverSampler(sampling_strategy='minority')
X_train, y_train = oversample.fit_resample(X, y)

gcv = GridSearchCV(base_model, param_grid = param_grid, cv = 3, scoring = 'f1')
gcv.fit(X_train, y_train)

hyperparameters = gcv.best_params_

hyper_name = "hyper.pkl"
with open(hyper_name, 'wb') as file:  
    pickle.dump(hyperparameters, file)
bucket = storage.Client().bucket(bucket_name)
blob = bucket.blob('{}/{}'.format("churn/metadata", hyper_name))
blob.upload_from_filename(hyper_name)  

grid_name = "gcv.pkl"
with open(grid_name, 'wb') as file:
    pickle.dump(gcv, file)
blob = bucket.blob('{}/{}'.format("churn/metadata", grid_name))
blob.upload_from_filename(grid_name)

