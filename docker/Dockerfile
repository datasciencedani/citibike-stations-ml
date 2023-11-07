FROM python:3.10-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ["predict.py", "model.joblib", "./"]

# Uncomment to run locally with Docker:
EXPOSE 8080
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8080", "predict:app"]

# # Uncomment to build container in GCP:
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 predict:app