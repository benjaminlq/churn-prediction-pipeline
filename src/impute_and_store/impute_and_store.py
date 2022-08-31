import argparse
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import pickle
import google.cloud.storage as storage

parser = argparse.ArgumentParser()
parser.add_argument(
    '--pre_impute_dataset', type=str, required=True, help='path of dataset input')
parser.add_argument(
    '--post_impute_dataset', type=str, required=True, help='path of dataset output')
parser.add_argument(
    '--bucket', type=str, required=True, help='GCS Bucket')

args = parser.parse_args()
pre_impute_dataset=args.pre_impute_dataset
post_impute_dataset=args.post_impute_dataset
bucket_name = args.bucket

def recode_TOT_OB_CALL_INTL_ROAM_CNT(data):
    if data <= 100:
        return("<=100")
    elif data <= 200:
        return("<=200")
    else:
        return(">200")

def recode_education(data):
    if data == " .":
        return("none")
    elif int(data) >= 3:
        return(">=3")
    else:
        return(data)

NUMERICAL_FEATURE_NAMES = [
    "SUBS_TENURE",
    "TOT_DAY_LAST_COMPLAINT_CNT",
    "TOT_DAY_LAST_SUSPENDED_CNT",
    "MTH_TO_SUBS_END_CNT",
    'REV_AMT_BASE_1',
    'REV_AMT_BASE_2',
    'CUST_AGE',
    'PCT_CHNG_IB_SMS_CNT'
]

EMBEDDING_CATEGORICAL_FEATURES = [
    "GENDER_CD",
    "EDUCATION_CD",
    "TOT_SRV_DROPPED_CNT",
    "TOT_OB_CALL_INTL_ROAM_CNT",
    'BARRING_REASON_CD', 
    "TOT_SRV_ADDED_CNT"]

TARGET_LABEL = ['CHURN_FLG']

pre_data = pd.read_csv(pre_impute_dataset + '.csv')
pre_data["EDUCATION_CD"] = pre_data["EDUCATION_CD"].apply(recode_education)
pre_data["TOT_OB_CALL_INTL_ROAM_CNT"] = pre_data["TOT_OB_CALL_INTL_ROAM_CNT"].apply(recode_TOT_OB_CALL_INTL_ROAM_CNT)
pre_data_cat = pre_data[EMBEDDING_CATEGORICAL_FEATURES]
pre_data_num = pre_data[NUMERICAL_FEATURE_NAMES]

bucket = storage.Client().bucket(bucket_name)

cat_imp = SimpleImputer(strategy = 'most_frequent')
num_imp = SimpleImputer(strategy = 'median')
imputed_data_cat = cat_imp.fit_transform(pre_data_cat)
imputed_data_num = num_imp.fit_transform(pre_data_num)

# Store Cat Imputer
cat_imp_name = "cat_imputer" + f".pkl"
with open(cat_imp_name, 'wb') as file:  
    pickle.dump(cat_imp, file)

blob = bucket.blob('{}/{}'.format("churn/artifact/preprocess",
                                cat_imp_name))
blob.upload_from_filename(cat_imp_name)

# Store Num Imputer
num_imp_name = "num_imputer" + f".pkl"
with open(num_imp_name, 'wb') as file:  
    pickle.dump(num_imp, file)

blob = bucket.blob('{}/{}'.format("churn/artifact/preprocess",
                                num_imp_name))
blob.upload_from_filename(num_imp_name)  

out_df = pd.DataFrame(np.concatenate((imputed_data_cat, imputed_data_num, pre_data[TARGET_LABEL].values),axis = 1),
                    columns = EMBEDDING_CATEGORICAL_FEATURES + NUMERICAL_FEATURE_NAMES + TARGET_LABEL)
out_df.to_csv(post_impute_dataset + ".csv" , index=False, encoding='utf-8-sig')

