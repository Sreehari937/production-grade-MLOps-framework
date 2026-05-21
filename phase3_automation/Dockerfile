FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PREFECT_API_URL=http://prefect:4200/api
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 4200

CMD ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"]