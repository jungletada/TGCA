#!/bin/bash
GPU=0
NODES=1
DATASET=COCO
DATACONFIG=data/MSCOCO/ImageLists

TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt

INPUTSIZE=448

SEGTRAINDIR=cam_mask_train
SEGVALDIR=cam_mask_val

MODELNAME=mctformerplus
WORKDIR=results_coco/${MODELNAME}

CUDA_VISIBLE_DEVICES=${GPU}

# # ============= Train Model ===============#
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_model_v2.py \
#     --dataset ${DATASET} \
#     --model ${MODELNAME} \
#     --train_list ${TRAINID} \
#     --work_space ${WORKDIR} \
#     --epoch 45 \
#     --batch_size 32 \
    
# ============= Make Class Activation Maps of Model=============#
python make_cam.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --work_space ${WORKDIR} \
    --cam_out_dir ${SEGTRAINDIR} \
    --train_list ${TRAINID} \
    --checkpoint ${WORKDIR}/${MODELNAME}_final.pth \
    # --checkpoint results_coco/mcta/mcta-deit-small-coco-4627.pth \
    
# ============= Evaluate Class Activation Maps =============#
python eval_cam_crf.py \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --eval_cam_dir ${SEGTRAINDIR} \
    --id_list ${TRAINID} \
    --curve_threshold \

