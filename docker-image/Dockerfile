FROM python:3.10.13-slim

WORKDIR /vision_api

COPY . /vision_api

RUN apt-get update

COPY ../requirements.txt ./
RUN pip install --trusted-host pypi.python.org -r requirements.txt
