
PROJECT_ID="churn-smu"
IMAGE_NAME="churn-data-endpoint_test"
TAG="latest"
GCR_IMAGE="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${TAG}"
echo 
docker build -t ${IMAGE_NAME} .
docker tag ${IMAGE_NAME} ${GCR_IMAGE}
docker push ${GCR_IMAGE}

