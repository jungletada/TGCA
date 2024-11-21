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
SEG_DIR=pseudo_mask
CRF_DIR=crf_mask

# MODELNAME=msgformer
# WORKDIR=results_voc/msgformer_${INPUTSIZE}

MODELNAME=mctformerplus
WORKDIR=results_voc/mctformerplus_448
CHECKPOINT=results_voc/mctformerplus_448/mctformerplus_6887.pth
# MODELNAME=mctnext
# WORKDIR=results_voc/mctnext_${INPUTSIZE}

CUDA_VISIBLE_DEVICES=${GPU}


# ============= Analysis Statistics of Model =============#
python hook_attn.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --work_space ${WORKDIR} \
    --train_list ${TRAINID} \
    --input_size ${INPUTSIZE} \
    --checkpoint ${CHECKPOINT} \
    # --checkpoint ${WORKDIR}/${MODELNAME}_best.pth \
    # --checkpoint ${WORKDIR}/msgformer_7388.pth \