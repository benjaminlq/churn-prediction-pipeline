import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    '--datapath', type=str, required=True, help='Name of the model file.')
parser.add_argument(
    '--dataset', type=str, required=True, help='GCS bucket name.')

args = parser.parse_args()
datapath=args.datapath
dataset=args.dataset

df_churn = pd.read_csv(datapath)

FEATURE_NAMES = ["GENDER_CD", "EDUCATION_CD", "TOT_SRV_DROPPED_CNT", "TOT_OB_CALL_INTL_ROAM_CNT",
                'BARRING_REASON_CD',  "TOT_SRV_ADDED_CNT", "SUBS_TENURE", "TOT_DAY_LAST_COMPLAINT_CNT",
                "TOT_DAY_LAST_SUSPENDED_CNT", "MTH_TO_SUBS_END_CNT", 'REV_AMT_BASE_1', 'REV_AMT_BASE_2',
                'CUST_AGE','PCT_CHNG_IB_SMS_CNT','CHURN_FLG']

df_churn = df_churn[FEATURE_NAMES]
df_churn.to_csv(dataset + ".csv", index=False, encoding = 'utf-8-sig')

