#!/bin/bash
# TODO: change the following accordingly for you.
#SBATCH --job-name="figer_predict"
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

lm_name=$1
encode_mode=$2
ckpt_dir=$3
orig_datadir=$4
datadir=$5
predict_fn=$6
predict_suff=$7
predict_thres=$8
flag1=${9}
flag2=${10}
flag3=${11}
flag4=${12}
flag5=${13}

# This is the script for training the Figer model.
# Path: classifier/train_script.sh

#{ mkdir -p "/disk/scratch/${datadir}"; datadir="/disk/scratch/${datadir}"; } || { mkdir -p "/disk/scratch1/${datadir}"; datadir="/disk/scratch1/${datadir}"; }

mkdir -p "${datadir}";
df -lh

cp -rv ../"${orig_datadir}"/"${predict_fn}".json"${predict_suff}".cache.h5 "${datadir}" || exit 1;

echo "Beginning Evaluation..."

python -u train.py --do_predict --data_dir "${datadir}" --predict_fn "${predict_fn}.json" \
--model_name_or_path ../../lms/"${lm_name}" --encode_mode "${encode_mode}" \
--output_dir "${ckpt_dir}" \
--predict_threshold "${predict_thres}" ${flag1} ${flag2} ${flag3} ${flag4} ${flag5} || exit 1;

mv "${datadir}/${predict_fn}_preds.json" ../"${orig_datadir}"/ || exit 1;

echo "Done."