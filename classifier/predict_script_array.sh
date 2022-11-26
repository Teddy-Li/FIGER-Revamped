#!/bin/bash
# TODO: change the following accordingly for you.
#SBATCH --job-name="figer_predict"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

lm_name=$1
encode_mode=$2
ckpt_dir=$3
datadir=$4
predict_fn=$5
label_set=$6
predict_thres=$7
flag1=${8}
flag2=${9}
flag3=${10}
flag4=${11}
flag5=${12}
cur_id=$SLURM_ARRAY_TASK_ID

# This is the script for training the Figer model.
# Path: classifier/train_script.sh

#{ mkdir -p "/disk/scratch/${datadir}"; datadir="/disk/scratch/${datadir}"; } || { mkdir -p "/disk/scratch1/${datadir}"; datadir="/disk/scratch1/${datadir}"; }

mkdir -p "${datadir}";
df -lh

predict_fn="${predict_fn}_${cur_id}"

cp -v "../news_data/${predict_fn}.json" "${datadir}/" || exit 1;

echo "Beginning Caching..."

python -u train.py --do_single_cache --data_dir "${datadir}" --predict_fn "${predict_fn}.json" \
--model_name_or_path ../../lms/"${lm_name}" --label_smoothing_factor 0.0 --labels_key "${label_set}" \
--reload_data ${flag1} ${flag2} ${flag3} ${flag4} ${flag5} || exit 1;

echo "Done Caching."
echo "Beginning Evaluation..."

python -u train.py --do_predict --data_dir "${datadir}" --predict_fn "${predict_fn}.json" \
--model_name_or_path ../../lms/"${lm_name}" --encode_mode "${encode_mode}" \
--output_dir "${ckpt_dir}" --predict_threshold "${predict_thres}" \
${flag1} ${flag2} ${flag3} ${flag4} ${flag5} || exit 1;

mv "${datadir}/${predict_fn}_preds.json" "../news_data/" || exit 1;
rm -v "${datadir}/${predict_fn}.json"
rm -v "${datadir}/${predict_fn}.json_${lm_name}_${label_set}_0.0.cache.h5"

echo "Done."