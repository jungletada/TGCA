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

MODELNAME=msgformer
WORKDIR=results_voc/msgformer_${INPUTSIZE}

# MODELNAME=mctformerplus
# WORKDIR=results_voc/mctformerplus_${INPUTSIZE}

# MODELNAME=mctnext
# WORKDIR=results_voc/mctnext_${INPUTSIZE}

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
    --seed 3 \
    --epoch 27 \
    --batch_per_gpu 19 \
    --cls_weight 3.0 \
    

# ============= Make Class Activation Maps of Model ============= #
python make_cam.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --work_space ${WORKDIR} \
    --train_list ${TRAINID} \
    --input_size ${INPUTSIZE} \
    --checkpoint ${WORKDIR}/${MODELNAME}_best.pth \
    # --checkpoint results_voc/msgformer_448/msgformer_7388.pth \

# ============= Evaluate Class Activation Maps =============#
python eval_cam.py \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --train_list ${TRAINID} \
    --alpha 1.35 \
    --curve_threshold \


# #============= Train and Infer Pixel Semantic Affnity =============
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_infer_psa.py \
#     --train True \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --train_list ${TRAINAUGID} \
#     --weights checkpoints/res38_cls.pth \
#     --seed 3 \
#     --low_alpha 1 \
#     --high_alpha 1.45 \



# #============= Infer with Pixel Semantic Affnity =============#
# python train_infer_psa.py --inference True \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --infer_list ${TRAINID} \
#     --seg_out_dir ${SEG_DIR} \
#     --threshold 0.47 \


# #============= Evaluate =============#
# python steps_voc/eval_sem_seg.py \
#     --work_space ${WORKDIR} \
#     --seg_out_dir ${SEG_DIR} \
#     --infer_list ${TRAINID} \
    #--seg_out_dir pseudo_mask_448 \

# # Save the generated mask to zip
# cd ${WORKDIR} && zip -r ${SEG_DIR}.zip ${SEG_DIR} && cd -


# # ============= Make Class Activation Maps of Model=============#
# python hook_attn.py \
#     --dataset ${DATASET} \
#     --model ${MODELNAME} \
#     --work_space ${WORKDIR} \
#     --train_list ${TRAINID} \
#     --input_size ${INPUTSIZE} \
#     --checkpoint ${WORKDIR}/mctformerplus_6887.pth \
    # --checkpoint ${WORKDIR}/${MODELNAME}_best.pth \
    # --checkpoint ${WORKDIR}/msgformer_7388.pth \