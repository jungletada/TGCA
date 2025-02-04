#!/bin/bash
DATASET=VOC12
DATACONFIG=data/VOCdevkit/VOC2012/ImageLists
TRAINAUGID=${DATACONFIG}/train_aug_id.txt
TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt

INPUTSIZE=448
MODELNAME=mcta
WORKDIR=results_voc/mcta

SEGTRAINDIR=cam_mask_train
SEGVALDIR=cam_mask_val
CUDA_VISIBLE_DEVICES=${GPU}

#============= Train Model ============= #
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
python train_model_v2.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --train_list ${TRAINAUGID} \
    --work_space ${WORKDIR} \
    --epochs 26 \
    --batch_size 19 \
    --seed 8 \
    --lr 1e-3 \
    
# ============= Make Class Activation Maps of Model ============= #
python make_cam.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --work_space ${WORKDIR} \
    --cam_out_dir ${SEGTRAINDIR} \
    --train_list ${TRAINID} \
    --input_size ${INPUTSIZE} \
    --checkpoint results_voc/mcta/mcta_best.pth \

# ============= Evaluate Class Activation Maps without CRF =============#
python eval_cam_crf.py \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --id_list ${TRAINID} \
    --eval_cam_dir ${SEGTRAINDIR} \
    --curve_threshold \

# # # ============= Make Class Activation Maps of Model ============= #
# # python make_cam.py \
# #     --dataset ${DATASET} \
# #     --model ${MODELNAME} \
# #     --work_space ${WORKDIR} \
# #     --train_list ${VALID} \
# #     --input_size ${INPUTSIZE} \
# #     --cam_out_dir ${SEGVALDIR} \
# #     --scales 1.0,0.75,1.25 \
# #     --checkpoint results_voc/mcta/mcta-deit-small-voc-7370.pth \

# # # ============= Evaluate Class Activation Maps No CRF =============#
# # python eval_cam_crf.py \
# #     --dataset ${DATASET} \
# #     --work_space ${WORKDIR} \
# #     --eval_cam_dir ${SEGVALDIR} \
# #     --id_list ${VALID} \
# #     --curve_threshold \