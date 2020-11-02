FROM python:3.7.3-stretch

WORKDIR /opt/app

# Install python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /opt/app

ENTRYPOINT ["python3.7", "server.py", "serve"]

# To build
# docker build -f Dockerfile  -t fastai-serving .

# To run
#docker run --rm -p 8501:8501 -t fastai-serving .

# To call the server
# (echo -n '{"data" : "'"$( base64 ~/Downloads/r1.jpg)"'"}') | curl -X POST -H "Content-Type: application/json" -d @- localhost:8501/analyze:predict