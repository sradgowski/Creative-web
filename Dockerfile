FROM python:3.8

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY src/main.py /src/main.py

ENTRYPOINT ["streamlit", "run"]

CMD ["/src/main.py"]