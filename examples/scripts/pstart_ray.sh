#!/bin/bash

MIP=$1
HOST_FILE=$2
PORT=6379
node_rank=$OMPI_COMM_WORLD_RANK

pkill -9 -f python
pkill -9 -f VLLM
ray stop --force || true

# Multi Node
if (( $(awk '{print $1}' ${HOST_FILE} | sort -u | wc -l) > 1 )); then
  if [[ "${node_rank}" == '0' ]]; then
    echo "start head ${node_rank}"
    ray start --head --node-ip-address ${MIP} --port=${PORT} --num-gpus=8
  else
    echo "start worker ${node_rank}"
    ray start --address="${MIP}:${PORT}" --num-gpus=8
  fi
fi
