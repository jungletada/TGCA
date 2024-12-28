#!/bin/bash
echo "  Available Dataset: VOC12, COCO";
echo "  |-- VOC12 dataset path: datasets/VOCdevkit/VOC2012";
echo "  |-- COCO dataset path: datasets/MSCOCO";

# NEED TO SET
GPU=0,1
NODES=2

DATASET=VOC12
DATACONFIG=configs/voc12

TRAINAUGID=${DATACONFIG}/train_aug_id.txt
TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt

INPUTSIZE=448
SEG_DIR=pseudo_mask_448
CRF_DIR=crf_mask

MODELNAME=msgformer
WORKDIR=results_voc/msgformer_${INPUTSIZE}

# MODELNAME=mctformerplus
# WORKDIR=results_voc/mctformerplus_${INPUTSIZE}

CUDA_VISIBLE_DEVICES=${GPU}

#============= Train Model ============= #
OMP_NUM_THREADS=${NODES} \
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    train_model.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --train_list ${TRAINAUGID} \
    --work_space ${WORKDIR} \
    --input_size ${INPUTSIZE} \
    --seed 8 \
    --epoch 35 \
    --batch_per_gpu 40 \


# # # ============= Make Class Activation Maps of Model=============#
# python visualize_cam.py \
#     --dataset ${DATASET} \
#     --model ${MODELNAME} \
#     --work_space ${WORKDIR} \
#     --train_list ${TRAINID} \
#     --input_size ${INPUTSIZE} \
#     --checkpoint results_voc/msgformer_448/msgformer_deit_small_s448.pth \