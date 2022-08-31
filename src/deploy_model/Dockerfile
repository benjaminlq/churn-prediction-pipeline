
FROM python:3.9

WORKDIR /deploy_model
COPY deploy_model.py deploy_model.py

RUN pip install google-cloud-aiplatform
RUN pip install sklearn
RUN pip install kfp

ENV PYTHONPATH="/check_model_metrics:"

