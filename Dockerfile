FROM python:3.6

EXPOSE 3000

RUN pip install -U pip

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

COPY ./deployment /app
WORKDIR app

ENTRYPOINT [ "streamlit", "run", "app.py" ]
