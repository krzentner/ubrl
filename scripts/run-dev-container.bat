docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --mount type=bind,source=.,target=/home/dev/outrl -it outrl
