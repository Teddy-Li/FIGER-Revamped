#!/bin/bash

#SBATCH --job-name="figer_predict"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

lm_name=$1
encode_mode=$2
ckpt_dir=$3
datadir=$4
predict_fn=$5
predict_suff=$6
predict_thres=$7
flag1=${8}
flag2=${9}
flag3=${10}
flag4=${11}
flag5=${12}

# This is the script for training the Figer model.
# Path: classifier/train_script.sh

mkdir -p "${datadir}";
df -lh

cp -rv ../json_data/"${predict_fn}".json"${predict_suff}".cache.h5 "${datadir}/" || exit 1;

echo "Beginning Evaluation..."

python -u train.py --do_predict --data_dir "${datadir}" --predict_fn "${predict_fn}.json" \
--model_name_or_path ../../lms/"${lm_name}" --encode_mode "${encode_mode}" \
--output_dir "${ckpt_dir}" --predict_threshold "${predict_thres}" \
${flag1} ${flag2} ${flag3} ${flag4} ${flag5} || exit 1;

mv "${datadir}/${predict_fn}_preds.json" "./logdir/" || exit 1;

echo "Done."