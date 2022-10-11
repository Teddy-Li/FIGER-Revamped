#!/bin/bash

#SBATCH --job-name="figer_cache"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

lm_name=$1
fn=$2
flag1=${3}
flag2=${4}
flag3=${5}
flag4=${6}
flag5=${7}

# This is the script for training the Figer model.
# Path: classifier/train_script.sh

echo "Beginning training..."

python -u train.py --do_infcache --predict_fn "${fn}" \
--model_name_or_path ../../lms/"${lm_name}" ${flag1} ${flag2} ${flag3} ${flag4} ${flag5}

echo "Done."