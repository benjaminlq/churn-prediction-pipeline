
FROM python:3.9

WORKDIR /digest
COPY data_ingest.py data_ingest.py

RUN pip install pandas
RUN pip install gcsfs

ENV PYTHONPATH="/digest:"

