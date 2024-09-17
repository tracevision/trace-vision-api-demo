FROM ubuntu:20.04


# Upgrade to Python 3.10
RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get install software-properties-common -y --no-install-recommends && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get -y update && \
    apt-get -y install python3.10 curl \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10  1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1


# Setting up pip for python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py > get-pip.py
RUN python3 ./get-pip.py "pip==22.3.1"
RUN rm get-pip.py
RUN pip3 install --root-user-action=ignore --upgrade setuptools
RUN pip3 install --root-user-action=ignore awscli==1.33.8
RUN pip3 install --root-user-action=ignore wheel


# Install ffmpeg
RUN apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            ffmpeg \
            && rm -rf /var/lib/apt/lists/*


# Install python requirements
ADD vision_api_demo_requirements.txt /tmp/req.txt
RUN pip3 install --root-user-action=ignore --trusted-host pypi.python.org -r /tmp/req.txt


WORKDIR /tmp
