FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install natten==0.17.1+torch220cu121 -f https://shi-labs.com/natten/wheels/

COPY . .

ENTRYPOINT ["bash"]