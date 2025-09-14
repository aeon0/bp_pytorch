# Boilerplate PyTorch

Easy project to get start with Docker, PyTorch, PyTorch Lightning, Hydra, Pydantic and VS Code development.

## Dependencies
- Docker: https://docs.docker.com/engine/install/ubuntu/
- VS Code: https://code.visualstudio.com/ (Extensions: Python, Dev Containers)
- Workstation with Nvidia GPU

## Get into the Container
In VS Code press ctrl+shift+P and search for "Dev Containers: Rebuild and Reopen in Container". The first time this will take a bit as Docker as to pull the Image. Once its ready you will be in the container and you will see only the workspace folder. Switch back with ctrl+shift+P and "Dev Containers: Reopen folder localy".

Note: I am poor and dont have an up-to-date Nvidia GPU, thus I am not using the latest PyTorch image. Switch out the image and adapt versions as you need. https://hub.docker.com/r/pytorch/pytorch/tags.

## Run and Debug Code
Run `python train.py` or `python predict.py` (need to have run training before) or run via f5.

## Issues
#### Container already in use
```
Error response from daemon: Conflict. The container name "/bp-pytorch-dev" is already in use by container "fbb94ded6b754f6629d3fb197e05f32d9cafd73f97ade0032e2edaa440da25db". You have to remove (or rename) that container to be able to reuse that name.
```
In case something crashed it can happen that the dev container has not shut down. In this case you have to manually stop it: `docker stop bp-pytorch-dev`