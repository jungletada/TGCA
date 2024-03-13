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
CUDA_VISIBLE_DEVICES=${GPU}
WORKDIR=results_voc/${MODEL}


# # ============= Train Model ===============#
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_model.py \
#     --dataset ${DATASET} \
#     --model deit_small_${MODEL} \
#     --work_space ${WORKDIR} \
#     --seed 1 \
#     --epoch 45 \
#     --batch_per_gpu 16 \
    

# # ============= Make Class Activation Maps of Model=============#
# python make_cam.py \
#     --dataset ${DATASET} \
#     --model deit_small_${MODEL} \
#     --work_space ${WORKDIR} \
#     --checkpoint ${WORKDIR}/deit_small_${MODEL}_best.pth \
#     --train_list configs/voc12/train_id.txt \


# # ============= Evaluate Class Activation Maps =============#
# python eval_cam.py \
#     --curve_threshold \
#     --work_space ${WORKDIR} \
#     --train_list configs/voc12/train_id.txt \


#============= Train and Infer Pixel Semantic Affnity =============#
OMP_NUM_THREADS=${NODES} \
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    train_infer_psa.py \
    --dataset VOC \
    --train True \
    --work_space ${WORKDIR} \
    --train_list configs/voc12/train_aug.txt \
    --weights checkpoints/res38_cls.pth \
    --seed 1 \
    --epoch 5 \
    --low_alpha 0.8 \
    --high_alpha 1.6 \


#============= Infer with Pixel Semantic Affnity =============#
python train_infer_psa.py \
    --dataset VOC \
    --inference True \
    --work_space ${WORKDIR} \
    --infer_list configs/voc12/train.txt \
    --seg_out_dir ${SEG_DIR} \
    --beta 11 \
    --logt 7 \
    --threshold 0.47 \


#============= Evaluate =============#
python steps_voc/eval_sem_seg.py \
    --work_space ${WORKDIR} \
    --seg_out_dir ${SEG_DIR} \


# # Save the generated mask to zip
# cd ${WORKDIR} && zip -r ${SEG_DIR}.zip ${SEG_DIR} && cd -

# CUDA_VISIBLE_DEVICES=0
# OMP_NUM_THREADS=1  \
#     torchrun \
#     --nproc_per_node=1 --nnodes=1  \
#     steps_voc/train_eval_seg.py \
#     --train True \
#     --seed 2 \
#     --num_epochs 30 \
#     --batch_per_gpu 4 \
#     --init_weights checkpoints/res38_cls.pth \


# python steps_voc/train_eval_seg.py \
#     --evaluate True \
#     --use_crf True \
#     --scales 0.5 0.75 1.0 1.25 1.5 \
#     # --pred_path val_ms \