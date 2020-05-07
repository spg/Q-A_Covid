# FROM python:3
FROM pytorch/pytorch:latest

WORKDIR /usr/src/app

COPY requirements.txt ./
COPY ./server ./
COPY ./server/weights ./weights
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8000

CMD [ "python", "server.py" ]
