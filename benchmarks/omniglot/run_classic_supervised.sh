#!/usr/bin/env bash

ADAPTATION=maml
BACKBONE=simple_cnn
DATASET=omniglot
NUM_CLASSES=5
NUM_SHOTS=1

BENCHMARK_TYPE=classic_supervised
DATA_DIR=$(realpath .)/data/${DATASET}

python -u -m meta_blocks.experiment.run \
  meta_blocks/adaptation=${BENCHMARK_TYPE}/${DATASET}/${ADAPTATION} \
  meta_blocks/benchmark=${BENCHMARK_TYPE}/${DATASET}/${NUM_CLASSES}way${NUM_SHOTS}shot \
  meta_blocks/data=${DATASET} \
  meta_blocks/network=${BENCHMARK_TYPE}/${DATASET}/${BACKBONE} \
  meta_blocks.data.source.data_dir=${DATA_DIR} \
  meta_blocks.compute.gpus.train.ids="0" \
  meta_blocks.compute.gpus.eval.ids="0" \