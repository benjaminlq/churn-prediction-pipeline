import argparse

import pandas as pd
import numpy as np
import pickle
import re
import google.cloud.storage as storage
from sklearn.preprocessing import OneHotEncoder, StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument(
    '--pre_enc_dataset', type=str, required=True, help='path of dataset input')
parser.add_argument(
    '--post_enc_dataset', type=str, required=True, help='path of dataset output')
parser.add_argument(
    '--bucket_name', type=str, required=True, help='GCS Bucket')

args = parser.parse_args()
pre_enc_dataset=args.pre_enc_dataset
post_enc_dataset=args.post_enc_dataset
bucket_name = args.bucket_name

NUMERICAL_FEATURE_NAMES = [
    "SUBS_TENURE",
    "TOT_DAY_LAST_COMPLAINT_CNT",
    "TOT_DAY_LAST_SUSPENDED_CNT",
    "MTH_TO_SUBS_END_CNT",
    'REV_AMT_BASE_1',
    'REV_AMT_BASE_2',
    'CUST_AGE',
    'PCT_CHNG_IB_SMS_CNT']

EMBEDDING_CATEGORICAL_FEATURES ={
    "x0_" :"GENDER_CD",
    "x1_" :"EDUCATION_CD",
    "x2_" :"TOT_SRV_DROPPED_CNT",
    "x3_" :"TOT_OB_CALL_INTL_ROAM_CNT",
    "x4_" :'BARRING_REASON_CD', 
    "x5_" :"TOT_SRV_ADDED_CNT"}

TARGET_LABEL = ['CHURN_FLG']

pre_data = pd.read_csv(pre_enc_dataset + '.csv')
pre_data_cat = pre_data[EMBEDDING_CATEGORICAL_FEATURES.values()]
pre_data_num = pre_data[NUMERICAL_FEATURE_NAMES]

bucket = storage.Client().bucket(bucket_name)

enc = OneHotEncoder()
scl = StandardScaler()

enc_data = enc.fit_transform(pre_data_cat)
enc_file_name = 'encoder' + f'.pkl'
with open(enc_file_name, 'wb') as file:  
    pickle.dump(enc, file)

blob = bucket.blob('{}/{}'.format("churn/artifact/preprocess", enc_file_name))
blob.upload_from_filename(enc_file_name)  

scl_data = scl.fit_transform(pre_data_num)
scl_file_name = 'scaler' + f'.pkl'
with open(scl_file_name, 'wb') as file:  
    pickle.dump(scl, file)

blob = bucket.blob('{}/{}'.format("churn/artifact/preprocess", scl_file_name))   
blob.upload_from_filename(scl_file_name) 

column_labels = list(enc.get_feature_names_out()) + NUMERICAL_FEATURE_NAMES + TARGET_LABEL
out_df = pd.DataFrame(np.concatenate((enc_data.toarray(), scl_data, pre_data[TARGET_LABEL].values),axis = 1),
                      columns = column_labels)
out_df.to_csv(post_enc_dataset + ".csv" , index=False, encoding='utf-8-sig')
    
