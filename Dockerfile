# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim
# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
# Copy local code to the container image.
ENV APP_HONE /app
WORKDIR ./
COPY . ./
# Install production dependencies.
RUN pip install Flask gunicorn
# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and & threads.
# For environments with multiple cPu cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to e to disable the timeouts of the workers to allow cloud Run to handle instance scaling.
#CMD ["exec", "gunicorn", "--bind", ":$PORT", "--workers", "1", "--threads", "8", "--timeout", "0", "main:app"]
CMD ["python", "main.py"]
