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

# MODELNAME=msgformer
# WORKDIR=results_voc/msgformer_${INPUTSIZE}

MODELNAME=simplevit
WORKDIR=results_voc/simplevit_${INPUTSIZE}

CUDA_VISIBLE_DEVICES=${GPU}


#============= Train Model ===============#
OMP_NUM_THREADS=${NODES} \
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    train_model.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --train_list ${TRAINAUGID} \
    --work_space ${WORKDIR} \
    --input_size ${INPUTSIZE} \
    --seed 8 \
    --epoch 27 \
    --batch_per_gpu 19 \
    

# # ============= Make Class Activation Maps of Model=============#
# python make_cam.py \
#     --dataset ${DATASET} \
#     --model ${MODELNAME} \
#     --work_space ${WORKDIR} \
#     --train_list ${TRAINID} \
#     --input_size ${INPUTSIZE} \
#     --checkpoint ${WORKDIR}/msgformer_deit_small_7323.pth \
    #--checkpoint ${WORKDIR}/${MODELNAME}_best.pth \
    

# # ============= Evaluate Class Activation Maps =============#
# python eval_cam.py \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --train_list ${TRAINID} \
#     --curve_threshold \


# #============= Train and Infer Pixel Semantic Affnity =============
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_infer_psa.py --train True \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --train_list ${TRAINAUGID} \
#     --seed 3 \
#     --low_alpha 0.35 \
#     --high_alpha 0.50 \
#     # --weights results_voc/resnet38d_448/ResNet38d_patch_224_last.pth \


# #============= Infer with Pixel Semantic Affnity =============#
# python train_infer_psa.py --inference True \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --infer_list ${TRAINAUGID} \
#     --seg_out_dir ${SEG_DIR} \
#     --threshold 0.46 \


# #============= Evaluate =============#
# python steps_voc/eval_sem_seg.py \
#     --work_space ${WORKDIR} \
#     --seg_out_dir ${SEG_DIR} \
#     #--seg_out_dir pseudo_mask_448 \

# # Save the generated mask to zip
# cd ${WORKDIR} && zip -r ${SEG_DIR}.zip ${SEG_DIR} && cd -