
import argparse
from google.cloud import storage
import tensorflow_data_validation as tfdv
import tensorflow_data_validation.statistics.stats_impl
import pandas as pd
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions, SetupOptions

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_data', type=str, required=True, help='input_data')
parser.add_argument(
    '--output_data', type=str, required=True, help='output_data')
parser.add_argument(
    '--project_id', type=str, required=True, help='project_id')
parser.add_argument(
    '--region', type=str, required=True, help='region')
parser.add_argument(
    '--gcs_temp_location', type=str, required=True, help='gcs_temp_location')
parser.add_argument(
    '--gcs_staging_location', type=str, required=True, help='gcs_staging_location')
parser.add_argument(
    '--bucket', type=str, required=True, help='bucket')

args = parser.parse_args()
input_data=args.input_data
project_id=args.project_id
region=args.region
output_data=args.output_data
gcs_temp_location=args.gcs_temp_location
gcs_staging_location=args.gcs_staging_location
bucket=args.bucket

job_name = 'dv4'
pre_data = pd.read_csv(input_data + '.csv')
schema_path = 'churn/metadata/schema/orig_stats.pb'

# Create and set your PipelineOptions.
options = PipelineOptions()

# For Cloud execution, set the Cloud Platform project, job_name,
# staging location, temp_location and specify DataflowRunner.
google_cloud_options = options.view_as(GoogleCloudOptions)
google_cloud_options.project = project_id
google_cloud_options.job_name = job_name
google_cloud_options.staging_location = gcs_staging_location
google_cloud_options.temp_location = gcs_temp_location
google_cloud_options.region = region
options.view_as(StandardOptions).runner = 'DataflowRunner'

setup_options = options.view_as(SetupOptions)
# PATH_TO_WHL_FILE should point to the downloaded tfdv wheel file.
setup_options.extra_packages = ['tensorflow_data_validation-1.8.0-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl']

storage_client = storage.Client()
storage_bucket = storage_client.bucket(bucket)
schema_exist = storage.Blob(bucket=storage_bucket, name=schema_path).exists(storage_client)

res = 'true'

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
    "EDUCATION_CD"]

if schema_exist:

    new_stats = tfdv.generate_statistics_from_csv(input_data.path + '.csv',
                                                  output_path=f'gs://{bucket}/churn/tmp/temp.pb',
                                                 pipeline_options=options,
                                                       )

    old_stats = tfdv.load_statistics(f'gs://{bucket}/churn/metadata/schema/orig_stats.pb')

    schema1 = tfdv.infer_schema(statistics=old_stats)
    for feature in NUMERICAL_FEATURE_NAMES:
        tfdv.get_feature(schema1, feature).drift_comparator.jensen_shannon_divergence.threshold = 0.15

    for feature in EMBEDDING_CATEGORICAL_FEATURES:
        tfdv.get_feature(schema1, feature).drift_comparator.infinity_norm.threshold = 0.1

    drift_anomalies = tfdv.validate_statistics(
        statistics=new_stats, schema=schema1, previous_statistics=old_stats)

    from google.protobuf.json_format import MessageToDict
    d = MessageToDict(drift_anomalies)
    val = d['driftSkewInfo'][0]['driftMeasurements'][0]['value']
    thresh = d['driftSkewInfo'][0]['driftMeasurements'][0]['threshold']

    if val < thresh:
        res = 'false'

if not schema_exist:    

    tfdv.generate_statistics_from_csv(input_data + '.csv',
                                      output_path=f'gs://{bucket}/churn/metadata/schema/orig_stats.pb',
                                      pipeline_options=options,
                                           )

assert res == 'true', "Data Validation failed"

