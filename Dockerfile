FROM python:3.10
ADD . /app
WORKDIR /app
RUN pip3 install -r requirements.txt --no-cache-dir
CMD exec gunicorn app:server --bind :$PORT