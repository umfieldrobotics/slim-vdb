#!/usr/bin/env bash

docker run -it \
    -e HOME \
    -v $HOME:$HOME \
    -v /path/to/SLIMVDB_DATASETS:/path/to/SLIMVDB_DATASETS \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --name slimvdb_docker \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    slimvdb_docker