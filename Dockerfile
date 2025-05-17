FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt update && apt install -y git curl net-tools tmux tree htop
RUN pip install --no-cache-dir boxx ipython
RUN mkdir -m 777 /.cache /.local /.config /.ipython
ENV PATH="/.local/bin:${PATH}"
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --no-cache-dir

ENTRYPOINT []
CMD ["/bin/bash"]

# Build docker image
# docker build --network=host -t diyer22/ddn .

# Run docker container
# docker run -it --gpus all --net=host -v `pwd`:/workspace --user $(id -u):$(id -g) diyer22/ddn python generate.py --debug 0 --batch=10 --seeds=0-99 --network weights/cifar-ddn.pkl
