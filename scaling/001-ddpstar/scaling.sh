#!/bin/bash

SCRIPTDIR_ABS=$(dirname "$0")

mkdir -p $SCRIPTDIR_ABS/log

/usr/bin/time -l .venv/bin/python "$SCRIPTDIR_ABS/launch.py" \
    $SCRIPTDIR_ABS 0 >> $SCRIPTDIR_ABS/log/run0.log 2>&1

/usr/bin/time -l .venv/bin/python "$SCRIPTDIR_ABS/launch.py" \
    $SCRIPTDIR_ABS 1 >> $SCRIPTDIR_ABS/log/run0.log 2>&1

/usr/bin/time -l .venv/bin/python "$SCRIPTDIR_ABS/launch.py" \
    $SCRIPTDIR_ABS 2 >> $SCRIPTDIR_ABS/log/run0.log 2>&1