#!/bin/bash
echo "  Available Dataset: VOC, COCO";
echo "  |-- VOC2012 dataset path: datasets/VOCdevkit/VOC2012";
echo "  |-- COCO dataset path: datasets/MSCOCO";

# NEED TO SET
GPU=0,1
NODES=2
MODEL=MCTG

CUDA_VISIBLE_DEVICES=${GPU}
OMP_NUM_THREADS=${NODES}
WORKDIR=results_voc/${MODEL}

#============= Train Pixel Semantic Affnity =============#
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    steps_voc/train_aff.py \
    --work_space ${WORKDIR} \
    --batch_per_gpu 4 \
    --seed 2 \
    --low_alpha 1 \
    --high_alpha 3 \
    --model_weights checkpoints/res38_cls.pth \


# #============= Infer with Pixel Semantic Affnity =============#
# python steps_voc/infer_aff.py \
#     --work_space ${WORKDIR} \
#     --checkpoint ${WORKDIR}/res38_aff_last.pth \
#     --infer_list configs/voc12/train.txt \
#     --cam_out_dir cam_mask \
#     --seg_out_dir pseudo_mask \
#     --threshold 0.41 \


# #================== Evaluate the pseudo mask ===================#
# python evaluation.py \
#     --work_space ${WORKDIR} \
#     --infer_list configs/voc12/train_id.txt \
#     --log_file eval_pseudo_mask.log \
#     --predict_dir ${WORKDIR}/pseudo_mask \
#     --type png \