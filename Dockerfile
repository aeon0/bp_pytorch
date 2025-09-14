FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

RUN pip install --no-cache-dir \
    pytorch-lightning==2.2.4 \
    torchmetrics==1.4.0 \
    hydra-core==1.3.2 \
    hydra_colorlog==1.2.0 \
    pydantic==2.11.9 \
    matplotlib==3.10.6 \
    tensorboard==2.20.0 \
    debugpy \
    ipython

WORKDIR /workspace
CMD ["/bin/bash"]
