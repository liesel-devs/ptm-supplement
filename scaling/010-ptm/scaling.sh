#!/bin/bash

SCRIPTDIR_ABS=$(dirname "$0")

# /usr/bin/time -l .venv/bin/python "$SCRIPTDIR_ABS/run.py" \
#     --warmup=1000 \
#     --posterior=2000 \
#     --testing=false \
#     --mcmc_strategy="iwls-nuts" \
#     --apply_jitter=false \
#     --data_type="mixture"\
#     --ntrain=2000 >> $SCRIPTDIR_ABS/log/run0.log 2>&1

/usr/bin/time -l .venv/bin/python "$SCRIPTDIR_ABS/run.py" \
    --warmup=1000 \
    --posterior=2000 \
    --testing=false \
    --mcmc_strategy="iwls-nuts" \
    --apply_jitter=false \
    --data_type="mixture"\
    --jobrow=1 \
    --ntrain=10000 >> $SCRIPTDIR_ABS/log/run1.log 2>&1

/usr/bin/time -l .venv/bin/python "$SCRIPTDIR_ABS/run.py" \
    --warmup=1000 \
    --posterior=2000 \
    --testing=false \
    --mcmc_strategy="iwls-nuts" \
    --apply_jitter=false \
    --data_type="mixture"\
    --jobrow=2 \
    --ntrain=20000 >> $SCRIPTDIR_ABS/log/run2.log 2>&1
