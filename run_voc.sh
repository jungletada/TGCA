#!/bin/bash
echo "  Available Dataset: VOC12, COCO";
echo "  |-- VOC12 dataset path: datasets/VOCdevkit/VOC2012";
echo "  |-- COCO dataset path: datasets/MSCOCO";

# NEED TO SET
GPU=0,1
NODES=2
DATASET=VOC12
MODEL=mctgformer
SEG_DIR=pseudo_mask
INPUTSIZE=448
CUDA_VISIBLE_DEVICES=${GPU}
WORKDIR=results_voc/${MODEL}_${INPUTSIZE}

DATACONFIG=configs/voc12
TRAINAUGID=${DATACONFIG}/train_aug_id.txt
TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt
TRAINAUGLIST=${DATACONFIG}/train_aug.txt
TRAINLIST=${DATACONFIG}/train.txt

# please note that mctformer(plus) use cls_weight=1.0

# # # ==================== Train Model ====================#
OMP_NUM_THREADS=${NODES} \
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    train_model.py \
    --dataset ${DATASET} \
    --model deit_small_${MODEL} \
    --work_space ${WORKDIR} \
    --input_size ${INPUTSIZE} \
    --seed 8 \
    --epoch 30 \
    --batch_per_gpu 16 \
    

# ============= Make Class Activation Maps of Model=============#
python make_cam.py \
    --dataset ${DATASET} \
    --model deit_small_${MODEL} \
    --work_space ${WORKDIR} \
    --train_list ${TRAINID} \
    --input_size ${INPUTSIZE} \
    --checkpoint ${WORKDIR}/deit_small_${MODEL}_best.pth \
    

# ============= Evaluate Class Activation Maps =============#
python eval_cam.py \
    --curve_threshold \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --train_list ${TRAINID} \


# #============= Train and Infer Pixel Semantic Affnity =============
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_infer_psa.py --train True \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --train_list ${TRAINAUGLIST} \
#     --weights checkpoints/res38_cls.pth \
#     --seed 3 \
#     --epoch 5 \
#     --low_alpha 0.55 \
#     --high_alpha 0.42 \


# #============= Infer with Pixel Semantic Affnity =============#
# python train_infer_psa.py --inference True \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --infer_list ${TRAINLIST} \
#     --seg_out_dir ${SEG_DIR} \
#     --beta 11 \
#     --logt 7 \
#     --threshold 0.45 \


# #============= Evaluate =============#
# python steps_voc/eval_sem_seg.py \
#     --work_space ${WORKDIR} \
#     --seg_out_dir ${SEG_DIR} \


# # Save the generated mask to zip
# cd ${WORKDIR} && zip -r ${SEG_DIR}.zip ${SEG_DIR} && cd -


# OMP_NUM_THREADS=${NODES} \
#     torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     steps_voc/train_eval_seg.py \
#     --work_space ${WORKDIR} \
#     --train True \
#     --seed 1 \
#     --num_epochs 30 \
#     --batch_per_gpu 2 \
#     --init_weights checkpoints/res38_cls.pth \


# python steps_voc/train_eval_seg.py \
#     --work_space ${WORKDIR} \
#     --evaluate True \
#     --use_crf True \
#     --scales 0.5 0.75 1.0 1.25 1.5 \
#     # --pred_path val_ms \