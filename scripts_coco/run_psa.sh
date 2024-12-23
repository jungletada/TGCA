#!/bin/bash
GPU=0,1
NODES=2
CUDA_VISIBLE_DEVICES=${GPU}
OMP_NUM_THREADS=${NODES} 

DATASET=COCO
DATACONFIG=data/MSCOCO/ImageLists
TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt

MODELNAME=mcta
WORKDIR=results_coco/mcta

CAMDIR=cam_mask_train
SEGDIR=pseudo_mask_train


#============= Train and Infer Pixel Semantic Affnity =============
OMP_NUM_THREADS=${NODES} \
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    train_infer_psa.py --train True \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --train_list ${TRAINID} \
    --cam_out_dir ${CAMDIR} \
    --batch_per_gpu 12 \
    --low_alpha 1 \
    --high_alpha 1.2 \
    

#============= Infer with Pixel Semantic Affnity =============#
python train_infer_psa.py --inference True \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --infer_list ${TRAINID} \
    --cam_out_dir ${CAMDIR} \
    --seg_out_dir ${SEGDIR} \
    --threshold 0.46 \


#============= Evaluate =============#
python steps_coco/eval_sem_seg.py \
    --work_space ${WORKDIR} \
    --seg_out_dir ${SEGDIR} \
    --infer_list ${TRAINID} \
    # --seg_out_dir pseudo_mask_448 \

# Save the generated mask to zip
# cd ${WORKDIR} && zip -r ${SEGDIR}.zip ${SEGDIR} && cd -