FROM python:3.13-trixie

COPY . /app/
COPY src/ /app/src/

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "gradio_app.py"]