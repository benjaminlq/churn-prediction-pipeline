
from google.cloud import aiplatform

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model', type=str, required=True, help='Model Path')
parser.add_argument(
    '--project', type=str, required=True, help='Project')
parser.add_argument(
    '--region', type=str, required=True, help='Region')
parser.add_argument(
    '--serving_container_image_uri', type=str, required=True, help='serving_container_image_uri')

args = parser.parse_args()

model = args.model
project=args.project
region = args.region
serving_container_image_uri = args.serving_container_image_uri

aiplatform.init(project = project, location = region)

DISPLAY_NAME = "churn_prediction_v2"
MODEL_NAME = "churn_rf_v2"
ENDPOINT_NAME = "churn_endpoint_v2"

def create_endpoint():
    endpoints = aiplatform.Endpoint.list(
        filter = 'display_name = "{}"'.format(ENDPOINT_NAME),
        order_by = 'create_time desc',
        project = project,
        location = region,
    )

    if len(endpoints) > 0:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(display_name = ENDPOINT_NAME,
                                              project = project,
                                              location = region)

    return endpoint

endpoint = create_endpoint()
endpoint_info = endpoint.resource_name.split('/')[-1]

model_upload = aiplatform.Model.upload(display_name = DISPLAY_NAME,
                                       artifact_uri = model.replace("model",""),
                                       serving_container_image_uri = serving_container_image_uri,
                                       serving_container_health_route=f'/v1/models/{MODEL_NAME}',
                                       serving_container_predict_route=f'/v1/models/{MODEL_NAME}:predict',
                                       serving_container_environment_variables = {"MODEL_NAME":MODEL_NAME,},
                                       )

model_deploy = model_upload.deploy(machine_type = "n1-standard-4",
                                  endpoint = endpoint,
                                  traffic_split = {"0":100},
                                  deployed_model_display_name = DISPLAY_NAME)
    
