#!/bin/bash

SCRIPTDIR_ABS=$(dirname "$0")

mkdir -p $SCRIPTDIR_ABS/log

/usr/bin/time -l .venv/bin/python "$SCRIPTDIR_ABS/run.py" \
    --warmup=1000 \
    --posterior=2000 \
    --data_type="mixture" \
    --ntrain=2000 \
    --jobdir=$SCRIPTDIR_ABS \
    --jobrow=3 >> $SCRIPTDIR_ABS/log/run3.log 2>&1

/usr/bin/time -l .venv/bin/python "$SCRIPTDIR_ABS/run.py" \
    --warmup=1000 \
    --posterior=2000 \
    --data_type="mixture" \
    --ntrain=10000 \
    --jobdir=$SCRIPTDIR_ABS \
    --jobrow=4 >> $SCRIPTDIR_ABS/log/run4.log 2>&1

/usr/bin/time -l .venv/bin/python "$SCRIPTDIR_ABS/run.py" \
    --warmup=1000 \
    --posterior=2000 \
    --data_type="mixture" \
    --ntrain=20000 \
    --jobdir=$SCRIPTDIR_ABS \
    --jobrow=5 >> $SCRIPTDIR_ABS/log/run5.log 2>&1
