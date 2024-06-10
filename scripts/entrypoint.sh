#!/bin/bash
cd ~/ubrl
source ~/venv/bin/activate
uv pip install -e .
if [ -z "$@" ]; then
  exec bash
else
  exec "$@"
fi
