
FROM python:3.12-slim

WORKDIR /app
COPY . /app
COPY prophet_model.pkl /app/prophet_model.pkl

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_ENV=production
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

##docker build -t flask-prophet-app .
