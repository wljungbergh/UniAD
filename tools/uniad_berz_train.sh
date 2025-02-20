#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --time 48:00:00
#SBATCH --output /proj/agp/users/%u/logs/%j.out
#SBATCH --account=berzelius-2023-365
#SBATCH --job-name=uniad_train
#

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
GPUS=8                                              #
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

export WANDB_ENTITY=wljungbergh

dos_path=/proj/cvl/users/x_willj/nuscenes_wazabi_bev_cache/doslo-nusc
uno_path=/proj/cvl/users/x_willj/nuscenes_wazabi_bev_cache/uno-nusc


if echo "$CFG" | grep -q "dos"; then
    cache_path=$dos_path
    name=dos
else
    cache_path=$uno_path
    name=uno
fi


WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$name-$T/

# see if 'stage2' in CFG
if echo "$CFG" | grep -q "stage2"; then
    suffix="stage2"
else
    suffix="stage1"
fi

MASTER_PORT=${MASTER_PORT:-28596}
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${PWD}/logs ]; then
    mkdir -p ${PWD}/logs
fi

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi




singularity exec --nv \
    --bind $PWD:/uniad \
    --bind /proj:/proj \
    --bind /proj/adas-data/data/nuscenes:/uniad/data/nuscenes \
    --bind $cache_path:/uniad/data/bev_cache \
    --pwd /uniad \
    --env PYTHONPATH="/uniad:${PYTHONPATH}" \
    --env WANDB_ENTITY=wljungbergh \
    --env WANDB_NAME=$name-uniad-$suffix \
    /proj/agp/containers/uniad-13-02-2024.sif \
    python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    tools/train.py \
    $CFG \
    --launcher pytorch ${@:4} \
    --work-dir ${WORK_DIR} \
    2>&1 | tee ${WORK_DIR}logs/train.$T