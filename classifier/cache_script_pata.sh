#!/bin/bash

lm_name=$1
fn=$2
lsf=$3
flag1=${4}
flag2=${5}
flag3=${6}
flag4=${7}
flag5=${8}

# This is the script for training the Figer model.
# Path: classifier/train_script.sh

echo "Beginning training..."

python -u train.py --do_single_cache --predict_fn "${fn}" --model_name_or_path ../../lms/"${lm_name}" \
--label_smoothing_factor "${lsf}" ${flag1} ${flag2} ${flag3} ${flag4} ${flag5}

echo "Done."