FROM python:3

RUN apt-get update && apt-get install -y \
	libsndfile1 \
	portaudio19-dev
RUN mkdir -p /root/src
COPY requirements.txt /root/src
WORKDIR /root/src

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt
