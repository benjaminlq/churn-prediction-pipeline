
import argparse
from google.cloud import aiplatform

parser = argparse.ArgumentParser()

parser.add_argument(
    '--endpoint', type=str, required=True, help='End Point ID')
parser.add_argument(
    '--project', type=str, required=True, help='Project')
parser.add_argument(
    '--region', type=str, required=True, help='Region')

args = parser.parse_args()

model = args.endpoint
project=args.project
region = args.region

aiplatform.init(project=project, location=location)

instance = [[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
             0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
             0.0, 0.0, -0.210, -0.695, 0.341, -0.528, 1.282, -1.632, 1.568, 1.309]]

endpoint = aiplatform.Endpoint(endpoint)

prediction = endpoint.predict(instances=instance)
assert int(prediction[0][0]) in [0, 1], "Prediction Error"
    
