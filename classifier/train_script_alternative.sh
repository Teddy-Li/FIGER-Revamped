#!/bin/bash

#SBATCH --job-name="figer_train"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

lm_name=$1
encode_mode=$2
lr=$3
lsf=$4
scrh_dir=$5
home_dir=$6
flag1=${7}
flag2=${8}
flag3=${9}
flag4=${10}
flag5=${11}

# This is the script for training the Figer model.
# Path: classifier/train_script.sh

mkdir -p "${scrh_dir}";
df -lh

rsync -avz "${home_dir}" "${scrh_dir}";
cd "${scrh_dir}" || exit;

echo "Beginning training..."

python -u train.py --do_train --model_name_or_path ../../lms/"${lm_name}" --encode_mode "${encode_mode}" --lr "${lr}" \
--num_train_epochs 5 --metric_best_model macro_f1 --label_smoothing_factor "${lsf}" \
--output_dir "${scrh_dir}/model_ckpts/" ${flag1} ${flag2} ${flag3} ${flag4} ${flag5}

echo "Training finished. Moving checkpoints to "

cd "${scrh_dir}" || exit;

rsync -avz "${scrh_dir}/model_ckpts" "${home_dir}/model_ckpts" && rm -rf "${scrh_dir}/model_ckpts";

echo "Done."