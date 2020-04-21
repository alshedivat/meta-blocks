#!/usr/bin/env bash

NUM_CLASSES=5
DATASET=omniglot
MODEL=feed_forward
BACKBONE=simple_cnn
DATA_DIR=$(realpath .)/data/${DATASET}

CUDA_VISIBLE_DEVICES=0 \
python -u -m meta_blocks.run \
  experiment=${DATASET}/${NUM_CLASSES}way/supervised/oracle \
  evaluation=${DATASET}/${NUM_CLASSES}way \
  dataset=${DATASET} \
  model=${DATASET}/${MODEL} \
  network=${DATASET}/${BACKBONE} \
  adaptation=maml \
  data.read_config.data_dir=${DATA_DIR}
