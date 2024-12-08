#!/bin/bash
GPU=0
NODES=1

DATASET=VOC12
DATACONFIG=data/VOCdevkit/VOC2012/ImageLists

TRAINAUGID=${DATACONFIG}/train_aug_id.txt
TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt

INPUTSIZE=448
MODELNAME=mcta
WORKDIR=results_voc/mcta

SEGDIR=cam_mask_train
CRFDIR=crf_mask_train

CUDA_VISIBLE_DEVICES=${GPU}

# #============= Train Model ============= #
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_model.py \
#     --dataset ${DATASET} \
#     --model ${MODELNAME} \
#     --train_list ${TRAINAUGID} \
#     --work_space ${WORKDIR} \
#     --seed 3 \
#     --epoch 30 \
#     --batch_per_gpu 40 \
    
# ============= Make Class Activation Maps of Model ============= #
python make_cam.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --work_space ${WORKDIR} \
    --cam_out_dir ${SEGDIR} \
    --train_list ${TRAINID} \
    --input_size ${INPUTSIZE} \
    --checkpoint results_voc/mcta/mcta-deit-small-voc-7458.pth
    # --checkpoint ${WORKDIR}/${MODELNAME}_best.pth \
    
# # ============= Evaluate Class Activation Maps without CRF =============#
# python eval_cam_crf.py \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --id_list ${TRAINID} \
#     --eval_cam_dir ${SEGDIR} \
#     --curve_threshold \


# ============= Evaluate Class Activation Maps =============#
python eval_cam_crf.py \
    --use_crf \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --eval_cam_dir ${SEGDIR} \
    --crf_cam_dir ${CRFDIR} \
    --alpha 1.1 \
    --id_list ${TRAIND} \

#============= Evaluate =============#
python steps_voc/eval_sem_seg.py \
    --work_space ${WORKDIR} \
    --seg_out_dir ${CRFDIR} \
    --infer_list ${TRAINID} \