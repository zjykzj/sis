FROM python:3.8
WORKDIR /Project/sis

COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD ["gunicorn", "server:app", "-c", "./gunicorn.conf.py"]