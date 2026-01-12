#!/bin/bash

SCRIPTDIR_ABS=$(dirname "$0")

/usr/bin/time -l .venv/bin/python "$SCRIPTDIR_ABS/run.py" --warmup=200 --posterior=2000 --thinning=1