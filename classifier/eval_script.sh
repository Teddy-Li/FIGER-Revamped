#!/bin/bash

#SBATCH --job-name="figer_eval"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

lm_name=$1
encode_mode=$2
ckpt_dir=$3
datadir=$4
flag1=${5}
flag2=${6}
flag3=${7}
flag4=${8}
flag5=${9}

# This is the script for training the Figer model.
# Path: classifier/train_script.sh

mkdir -p "${datadir}";
df -lh

cp -rv ../json_data/dev*.cache.h5 "${datadir}";
cp -rv ../json_data/test*.cache.h5 "${datadir}";

echo "Beginning Evaluation..."

python -u train.py --do_dev --data_dir "${datadir}" --model_name_or_path ../../lms/"${lm_name}" --encode_mode "${encode_mode}" \
--output_dir "${ckpt_dir}" ${flag1} ${flag2} ${flag3} ${flag4} ${flag5}


echo "Done."