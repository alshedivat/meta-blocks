#!/usr/bin/env bash

ADAPTATION=reptile
BACKBONE=simple_cnn
DATASET=omniglot
MODEL=feed_forward
NUM_CLASSES=5

BENCHMARK_TYPE=classic_supervised
DATA_DIR=$(realpath .)/data/${DATASET}

CUDA_VISIBLE_DEVICES=0 \
python -u run.py \
  meta_blocks/adaptation=${BENCHMARK_TYPE}/${DATASET}/${NUM_CLASSES}way/${ADAPTATION} \
  meta_blocks/benchmark=${BENCHMARK_TYPE}/${DATASET}/${NUM_CLASSES}way \
  meta_blocks/data=${DATASET} \
  meta_blocks/model=${MODEL} \
  meta_blocks/network=${BENCHMARK_TYPE}/${DATASET}/${BACKBONE} \
  meta_blocks.data.read_config.data_dir=${DATA_DIR} \
