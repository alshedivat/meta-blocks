#!/bin/bash
#
# Fetch Mini-ImageNet.
#

WORK_DIR=$(pwd)
TARGET_DIR="data/miniimagenet"
IMAGENET_TERMS=$(cat imagenet_terms.txt)

# Exit if any command fails.
set -e

echo "MiniImageNet is a subset of the ImageNet dataset and is only allowed to be " \
     "downloaded by researchers for non-commercial research and educational purposes."
echo "To download the data, you need to agree to the terms of ImageNet:"
echo "---"
echo "$IMAGENET_TERMS"
echo "---"

read -r -p "Do you agree to follow the terms? [y/N] " response
if [[ ! $response =~ ^([yY][eE][sS]|[yY])+$ ]]
then
   echo "Aborting."
   [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
fi

if [ ! -d data ]; then
    mkdir data
fi

if [ ! -d "$TARGET_DIR" ]; then
  mkdir "$TARGET_DIR"
  cd "$TARGET_DIR"
  echo "Fetching miniImageNet..."
  for set in "train" "valid" "test"; do
    wget "https://www.cs.cmu.edu/~mshediva/data/miniimagent/$set.tar"
    tar -xf "$set.tar"
    rm "$set.tar"
  done
  # Rename val to valid.
  mv val/ valid/
  cd "$WORK_DIR"
  echo "Done."
else
  echo "miniImageNet has been already been fetched."
fi
