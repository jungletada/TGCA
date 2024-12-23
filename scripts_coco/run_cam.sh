#!/bin/bash
GPU=0,1
NODES=2
DATASET=COCO
DATACONFIG=data/MSCOCO/ImageLists

TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt

INPUTSIZE=448

SEGDIR=cam_mask_train
CRFDIR=crf_mask_train

MODELNAME=mcta
WORKDIR=results_coco/mcta

CUDA_VISIBLE_DEVICES=${GPU}

# # ============= Train Model ===============#
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_model.py \
#     --dataset ${DATASET} \
#     --model ${MODELNAME} \
#     --train_list ${TRAINID} \
#     --work_space ${WORKDIR} \
#     --seed 8 \
#     --epoch 35 \
#     --batch_per_gpu 20 \
    
# ============= Make Class Activation Maps of Model=============#
python make_cam.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --work_space ${WORKDIR} \
    --cam_out_dir ${SEGDIR} \
    --train_list ${TRAINID} \
    --checkpoint results_coco/mcta/mcta-deit-small-coco-4627.pth \
    # --checkpoint ${WORKDIR}/${MODELNAME}_best.pth \
    
    
# ============= Evaluate Class Activation Maps =============#
python eval_cam_crf.py \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --eval_cam_dir ${SEGDIR} \
    --id_list ${TRAINID} \
    --curve_threshold \