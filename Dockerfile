FROM nvcr.io/nvidia/tensorrt:22.11-py3

RUN pip install --upgrade pip && pip install --upgrade tensorrt

RUN git clone https://github.com/NVIDIA/TensorRT.git -b release/8.5 --single-branch \
    && cd TensorRT/ \
    && git submodule update --init --recursive

ENV TRT_OSSPATH=/workspace/TensorRT
WORKDIR /workspace/TensorRT

RUN mkdir -p build \
    && cd build \
    && cmake .. -DTRT_OUT_DIR=$PWD/out \
    && cd plugin \
    && make -j$(nproc)

ENV PLUGIN_LIBS="${TRT_OSSPATH}/build/out/libnvinfer_plugin.so"

WORKDIR /workspace

# Install git
RUN apt-get update && apt-get install -y git
ENV HF_AUTH_TOKEN=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
ENV MODEL_NAME=runwayml/stable-diffusion-v1-5
# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ADD . .

RUN python3 download.py

EXPOSE 8000

CMD python3 -u server.py
