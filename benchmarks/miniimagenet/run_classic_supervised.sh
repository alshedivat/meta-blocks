#!/usr/bin/env bash

METHOD=maml
BACKBONE=simple_cnn
DATASET=miniimagenet
NUM_CLASSES=5
NUM_SHOTS=1

BENCHMARK_TYPE=classic_supervised
DATA_DIR=$(realpath .)/data/${DATASET}

python -u run.py \
  benchmark=${BENCHMARK_TYPE}/${NUM_CLASSES}way/${NUM_SHOTS}shot \
  backbone=${BENCHMARK_TYPE}/${NUM_CLASSES}way/${NUM_SHOTS}shot/${BACKBONE} \
  method=${BENCHMARK_TYPE}/${NUM_CLASSES}way/${NUM_SHOTS}shot/${METHOD} \
  meta_blocks.data.source.data_dir=${DATA_DIR} \
  meta_blocks/compute=2gpu \
