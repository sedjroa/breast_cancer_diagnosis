FROM docker.io/python:3.10-slim

WORKDIR /stream_app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY /app .
COPY /assets .
COPY /data .
COPY /model .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]