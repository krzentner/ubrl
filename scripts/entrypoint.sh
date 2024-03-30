#!/bin/bash
cd ~/outrl
source ~/venv/bin/activate
pip install -e './stick[recommended]'
pip install -e .
if [ -z "$@" ]; then
  exec bash
else
  exec "$@"
fi
