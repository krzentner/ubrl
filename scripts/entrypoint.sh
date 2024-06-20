#!/bin/bash
cd ~/ubrl
source ~/venv/bin/activate
echo "Using python: $(which python)"
uv pip install --no-deps --editable .
if [ -z "$@" ]; then
  exec bash
else
  exec "$@"
fi
