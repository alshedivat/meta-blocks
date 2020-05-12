#!/usr/bin/env bash

METHOD=maml
BACKBONE=simple_cnn
DATASET=miniimagenet
NUM_CLASSES=5

BENCHMARK_TYPE=self_supervised
TASK_DISTRIBUTION=umtra
DATA_DIR=$(realpath .)/data/${DATASET}

python -u run.py \
  benchmark=${BENCHMARK_TYPE}/${NUM_CLASSES}way/${TASK_DISTRIBUTION} \
  backbone=${BENCHMARK_TYPE}/${NUM_CLASSES}way/${TASK_DISTRIBUTION}/${BACKBONE} \
  method=${BENCHMARK_TYPE}/${NUM_CLASSES}way/${TASK_DISTRIBUTION}/${METHOD} \
  meta_blocks.data.source.data_dir=${DATA_DIR} \
  meta_blocks/compute=2gpu \
