#!/bin/bash
cd ~/outrl
source ~/venv/bin/activate
uv pip install -e .
if [ -z "$@" ]; then
  exec bash
else
  exec "$@"
fi
