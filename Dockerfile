FROM python:3.9

COPY ./ /app
WORKDIR /app
RUN apt update && apt install build-essential
RUN pip install --upgrade pip && pip install --no-cache-dir --upgrade -r requirements.txt
CMD ["python", "./flwr/torch_vision/server.py"]