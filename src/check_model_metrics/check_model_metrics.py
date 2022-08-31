
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument(
    '--base_metrics', type=str, required=True, help='Base Metrics')
parser.add_argument(
    '--threshold_dict', type=str, required=True, help='Threshold F1-Score')
parser.add_argument(
    '--deploy_path', type=str, required=True, help='Deploy Status')

args = parser.parse_args()

base_metrics = args.base_metrics
threshold_dict=args.threshold_dict
deploy_path = args.deploy_path

def threshold_check(val1,val2):
    cond = "False"
    if val1 > val2:
        cond = "True"
    return cond

thresholds_dict = json.loads(threshold_dict)
deploy = threshold_check(float(base_metrics), thresholds_dict['f1_score'])

with open(deploy_path + '.txt', 'w') as file:
    file.write(deploy)
    
