FROM python:3.10.12-bookworm
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
COPY . /code/
WORKDIR /code
RUN python3 setup.py install

