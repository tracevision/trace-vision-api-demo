FROM python:3.10.13-slim

WORKDIR /vision_api

COPY . /vision_api

RUN apt-get update

RUN pip install --trusted-host pypi.python.org -r /vision_api/requirements.txt
