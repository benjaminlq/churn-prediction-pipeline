
FROM python:3.9

WORKDIR /endpoint_test
COPY endpoint_test.py check_model_metrics.py

RUN pip install google-cloud-aiplatform

ENV PYTHONPATH="/endpoint_test:"

