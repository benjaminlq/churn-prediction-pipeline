FROM gcr.io/churn-smu/churn-tfdv:latest
WORKDIR /tfdv
COPY tfdv.py tfdv.py

RUN pip install pandas
RUN pip install gcsfs
RUN pip install google-cloud-storage

ENV PYTHONPATH="/digest:"
