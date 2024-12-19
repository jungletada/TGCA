#!/bin/bash
GPU=0
NODES=1

DATASET=VOC12
DATACONFIG=data/VOCdevkit/VOC2012/ImageLists

TRAINAUGID=${DATACONFIG}/train_aug_id.txt
TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt

INPUTSIZE=448
CAMDIR=cam_mask_train
SEGDIR=pseudo_mask_448

MODELNAME=mcta
WORKDIR=results_voc/mcta

CUDA_VISIBLE_DEVICES=${GPU}

# #============= Train and Infer Pixel Semantic Affnity =============
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_infer_psa.py \
#     --train True \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --cam_out_dir ${CAMDIR} \
#     --train_list ${TRAINAUGID} \
#     --weights checkpoints/res38_cls.pth \


# #============= Infer with Pixel Semantic Affnity =============#
# python train_infer_psa.py \
#     --inference True \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --infer_list ${TRAINAUGID} \
#     --cam_out_dir ${CAMDIR} \
#     --seg_out_dir ${SEGDIR} \
#     --threshold 0.48 \

#============= Evaluate =============#
python steps_voc/eval_sem_seg.py \
    --work_space ${WORKDIR} \
    --seg_out_dir ${SEGDIR} \
    --infer_list ${TRAINID} \

# # Save the generated mask to zip
# cd ${WORKDIR} && zip -r ${SEGDIR}.zip ${SEGDIR} && cd -