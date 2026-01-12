#!/bin/bash

SCRIPTDIR_ABS=$(dirname "$0")

/usr/bin/time -l .venv/bin/python "$SCRIPTDIR_ABS/run.py" --warmup=1000 --posterior=200 --testing=true --mcmc_strategy="iwls-iwls_fixed" --apply_jitter=false