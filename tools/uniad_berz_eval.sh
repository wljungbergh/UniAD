#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --time 10:00:00
#SBATCH --output /proj/agp/users/%u/logs/%j.out
#SBATCH --account=berzelius-2023-365
#SBATCH --job-name=uniad_eval
#

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=8                                             #
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

if echo "$CFG" | grep -q "dos"; then
    cache_path=/proj/cvl/users/x_willj/nuscenes_wazabi_bev_cache/doslo-nusc
    name=dos
else
    cache_path=/proj/cvl/users/x_willj/nuscenes_wazabi_bev_cache/uno-nusc
    name=uno
fi

MASTER_PORT=${MASTER_PORT:-28596}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/eval-$name-$T/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${PWD}/logs ]; then
    mkdir -p ${PWD}/logs
fi

singularity exec --nv \
    --bind $PWD:/uniad \
    --bind /proj:/proj \
    --bind /proj/adas-data/data/nuscenes:/uniad/data/nuscenes \
    --bind $cache_path:/uniad/data/bev_cache \
    --pwd /uniad \
    --env WANDB_ENTITY=wljungbergh \
    --env WANDB_NAME=$name-uniad-eval \
    --env PYTHONPATH="/uniad:${PYTHONPATH}" \
    /proj/agp/containers/uniad-13-02-2024.sif \
    python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    tools/test.py \
    $CFG \
    $CKPT \
    --out $WORK_DIR/results.pkl \
    --launcher pytorch ${@:4} \
    --eval bbox
