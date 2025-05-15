FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt update && apt install -y git curl net-tools tmux tree htop
RUN pip install --no-cache-dir boxx ipython
RUN mkdir -m 777 /.cache
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt --no-cache-dir

ENTRYPOINT []
CMD ["/bin/bash"]

# build docker image
# docker build --network=host -t diyer22/ddn .
# 
# run docker container
# docker run -it --gpus all --net=host -v `pwd`:/workspace -v /home/yl/ws/ddn/asset/v15-00023-ffhq-64x64-blockn64_outputk64_chain.dropout0.05-shot-117913.pkl:/share/ddn.pkl -v /tmp/share:/share diyer22/ddn python generate.py --debug 0 --batch=3 --seeds=0-15 --network /share/ddn.pkl
# 