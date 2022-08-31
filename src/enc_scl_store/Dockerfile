
FROM python:3.9

WORKDIR /enc_scl_store
COPY enc_scl_store.py enc_scl_store.py

RUN pip install pandas
RUN pip install sklearn
RUN pip install numpy
RUN pip install --upgrade google-api-python-client
RUN pip install google-cloud-storage
RUN pip install gcsfs

ENV PYTHONPATH="/enc_scl_store:"

