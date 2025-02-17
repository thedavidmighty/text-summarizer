FROM python:3.10-slim

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install --upgrade accelerate
RUN pip install transformers accelerate && pip cache purge

CMD ["python3", "app.py"]
