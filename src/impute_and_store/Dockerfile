
FROM python:3.9

WORKDIR /impute_store
COPY impute_and_store.py impute_and_store.py

RUN pip install pandas
RUN pip install sklearn
RUN pip install numpy
RUN pip install --upgrade google-api-python-client
RUN pip install google-cloud-storage
RUN pip install gcsfs

ENV PYTHONPATH="/impute_store:"

