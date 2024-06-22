FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
WORKDIR /app
RUN apt update -q && apt install -fyqq git
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install .
ENTRYPOINT ["sleep", "inf"]
