#!/bin/bash

#SBATCH --job-name="figer_train"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

lm_name=$1
encode_mode=$2
lr=$3
lsf=$4
datadir=$5
outdir=$6
out_dest=$7
flag1=${8}
flag2=${9}
flag3=${10}
flag4=${11}
flag5=${12}

# This is the script for training the Figer model.
# Path: classifier/train_script.sh

mkdir -p "${outdir}";
mkdir -p "${datadir}";

df -lh

rsync -avz ../json_data/*.cache.h5 "${datadir}";

echo "Beginning training..."

python -u train.py --do_train --data_dir "${datadir}" --model_name_or_path ../../lms/"${lm_name}" --encode_mode "${encode_mode}" --lr "${lr}" \
--num_train_epochs 5 --metric_best_model macro_f1 --label_smoothing_factor "${lsf}" \
--output_dir "${outdir}" ${flag1} ${flag2} ${flag3} ${flag4} ${flag5}

echo "Training finished. Moving checkpoints to "

cp -rv "${outdir}" "${out_dest}" && rm -rf "${outdir}";

echo "Done."