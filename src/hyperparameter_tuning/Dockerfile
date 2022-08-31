
FROM python:3.9

WORKDIR /hyperparameter_tuning
COPY hyperparameter_tuning.py hyperparameter_tuning.py

RUN pip install pandas
RUN pip install sklearn
RUN pip install numpy
RUN pip install --upgrade google-api-python-client
RUN pip install google-cloud-storage
RUN pip install gcsfs
RUN pip install -U imbalanced-learn

ENV PYTHONPATH="/enc_scl_store:"

