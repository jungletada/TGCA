#!/bin/bash
echo "  Available Dataset: VOC12, COCO";
echo "  |-- VOC12 dataset path: datasets/VOCdevkit/VOC2012";
echo "  |-- COCO dataset path: datasets/MSCOCO";

# NEED TO SET
GPU=0,1
NODES=2
DATASET=COCO
MODEL=mctgformer
MODELNAME=deit_small_mctgformer
SEGDIR=pseudo_mask
INPUTSIZE=448

CUDA_VISIBLE_DEVICES=${GPU}
WORKDIR=results_coco/${MODEL}_${INPUTSIZE}

DATACONFIG=configs/coco
TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt



# ============= Train Model ===============#
OMP_NUM_THREADS=${NODES} \
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    train_model.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --train_list ${TRAINID} \
    --work_space ${WORKDIR} \
    --input_size ${INPUTSIZE} \
    --seed 3 \
    --epoch 30 \
    --batch_per_gpu 20 \
    

# ============= Make Class Activation Maps of Model=============#
python make_cam.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --work_space ${WORKDIR} \
    --train_list ${TRAINID} \
    --input_size ${INPUTSIZE} \
    --checkpoint ${WORKDIR}/${MODELNAME}_best.pth \
    

# ============= Evaluate Class Activation Maps =============#
python eval_cam.py \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --train_list ${TRAINID} \
    --curve_threshold \


# #============= Train and Infer Pixel Semantic Affnity =============
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_infer_psa.py --train True \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --train_list ${TRAINID} \
#     --weights checkpoints/res38_cls.pth \
#     --batch_per_gpu 8 \
#     --seed 3 \
#     --epoch 5 \
#     --low_alpha 0.38 \
#     --high_alpha 0.48 \


# #============= Infer with Pixel Semantic Affnity =============#
# python train_infer_psa.py --inference True \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --infer_list ${TRAINID} \
#     --seg_out_dir ${SEGDIR} \
#     --beta 11 \
#     --logt 7 \
#     --threshold 0.45 \


# #============= Evaluate =============#
# python steps_coco/eval_sem_seg.py \
#     --work_space ${WORKDIR} \
#     --seg_out_dir ${SEGDIR} \


# # Save the generated mask to zip
# cd ${WORKDIR} && zip -r ${SEG_DIR}.zip ${SEG_DIR} && cd -