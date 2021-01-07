#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$3 --master_port=$4 $(dirname "$0")/online_inference_recognizer.py $1 $2 --launcher pytorch ${@:5}
