import argparse

import operator, pickle, json
import pandas as pd
import numpy as np
import google.cloud.storage as storage
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import roc_curve, confusion_matrix, f1_score
import gcsfs

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, required=True, help='path of dataset input')
parser.add_argument(
    '--bucket_name', type=str, required=True, help='GCS Bucket')
parser.add_argument(
    '--model', type=str, required=True, help='Model Artifact Path')
parser.add_argument(
    '--classification_metrics', type=str, required=True, help='Confusion Matrix')
parser.add_argument(
    '--base_metrics', type=str, required=True, help='Base Metrics')
parser.add_argument(
    '--feature_importance', type=str, required=True, help='Feature Importance Table')

args = parser.parse_args()
dataset=args.dataset
bucket_name = args.bucket_name
model=args.model
classification_metrics = args.classification_metrics
base_metrics=args.base_metrics
feature_importance = args.feature_importance

df = pd.read_csv(dataset + ".csv")
feature_label = df.columns[:-1]
data = df.values

X = data[:,:-1]
y = data[:,-1].astype(int)

X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state = 2022)
oversample = RandomOverSampler(sampling_strategy='minority')
X_train, y_train = oversample.fit_resample(X_train, y_train)

fs = gcsfs.GCSFileSystem(project='practice-smu-123')
fs.ls(bucket_name)

bucket = storage.Client().bucket(bucket_name)
blob = bucket.blob("churn/metadata/hyper.pkl")
hyper_name = "hyper.pkl"
blob.download_to_filename(hyper_name)
file = open(hyper_name, 'rb')
hyperparameters = pickle.load(file)
file.close()

model_rf = RandomForestClassifier(n_estimators = 20)
model_rf.set_params(**hyperparameters)
model_rf.fit(X_train, y_train)
feature_importances = model_rf.feature_importances_
rf_feature_importance = {feature_label[i] : model_rf.feature_importances_[i]
                            for i in range(len(model_rf.feature_importances_))}
rf_feature_importance = dict(sorted(rf_feature_importance.items(), reverse = True,
                                    key=operator.itemgetter(1)))
feature_importance_df = pd.DataFrame([rf_feature_importance.keys(),rf_feature_importance.values()],
                                    index = ["Feature","Importance"]).transpose()
feature_importance_df.to_csv(feature_importance + ".csv", index=False,
                             header = False, encoding='utf-8-sig')


# blob = bucket.blob('{}/{}'.format("churn/metadata", "feature_importance.csv"))
# blob.upload_from_filename(feature_importance + ".csv")

file_name = model + f'.pkl'
with fs.open(file_name, 'wb') as file:
    pickle.dump(model_rf, file)
    
y_preds = model_rf.predict(X_test)

y_scores = model_rf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_true = y_test, y_score = y_scores, pos_label = True)
classification_metrics.log_roc_curve(fpr.tolist(), tpr.tolist(),thresholds.tolist())

classification_metrics.log_confusion_matrix(["False","True"],confusion_matrix(y_test, y_preds).tolist())

f1_score = f1_score(y_test, y_preds)
#  thresholds_dict = json.loads(thresholds_dict_str)
model.metadata["frameword"] = "rf"
model.metadata["f1_score"] = float(f1_score)
base_metrics.log_metric("f1_score",float(f1_score))

