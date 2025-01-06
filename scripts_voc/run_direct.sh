#!/bin/bash
GPU=0,1
NODES=2

DATASET=VOC12
DATACONFIG=data/VOCdevkit/VOC2012/ImageLists
TRAINAUGID=${DATACONFIG}/train_aug_id.txt
TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt

INPUTSIZE=448
MODELNAME=mctformerplus
WORKDIR=results_voc/mctformerplus

SEGDIR=cam_mask_val
CRFDIR=crf_mask_val

CUDA_VISIBLE_DEVICES=${GPU}

# ============= Make Class Activation Maps of Model ============= #
python make_cam.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --work_space ${WORKDIR} \
    --train_list ${VALID} \
    --input_size ${INPUTSIZE} \
    --cam_out_dir ${SEGDIR} \
    --scales 1.0,0.75,1.25 \
    --checkpoint results_voc/mctformerplus/mctformerplus_6887.pth \

# ============= Evaluate Class Activation Maps No CRF =============#
python eval_cam_crf.py \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --eval_cam_dir ${SEGDIR} \
    --id_list ${VALID} \
    --curve_threshold \

# ============= Evaluate Class Activation Maps =============#
python eval_cam_crf.py \
    --use_crf \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --eval_cam_dir ${SEGDIR} \
    --crf_cam_dir ${CRFDIR} \
    --alpha 1.15 \
    --id_list ${VALID} \

#============= Evaluate =============#
python steps_voc/eval_sem_seg.py \
    --work_space ${WORKDIR} \
    --seg_out_dir ${CRFDIR} \
    --infer_list ${VALID} \

# # Save the generated mask to zip
# cd ${WORKDIR} && zip -r ${CRFDIR}.zip ${CRFDIR} && cd -