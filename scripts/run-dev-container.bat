docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --publish 6006:6006 --mount type=bind,source=.,target=/home/dev/ubrl -it ubrl-dev
