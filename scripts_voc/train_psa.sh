#!/bin/bash
echo "  Available Dataset: VOC, COCO";
echo "  |-- VOC2012 dataset path: datasets/VOCdevkit/VOC2012";
echo "  |-- COCO dataset path: datasets/MSCOCO";

# NEED TO SET
GPU=0,1
NODES=2
MODEL=MCTG
SEG_DIR=pseudo_mask
CUDA_VISIBLE_DEVICES=${GPU}
OMP_NUM_THREADS=${NODES}
WORKDIR=results_voc/${MODEL}

# #============= Train Pixel Semantic Affnity =============#
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     steps_voc/train_aff.py \
#     --work_space ${WORKDIR} \
#     --batch_per_gpu 4 \
#     --seed 2 \
#     --low_alpha 1 \
#     --high_alpha 3 \
#     --model_weights checkpoints/res38_cls.pth \


# #============= Infer with Pixel Semantic Affnity =============#
# python steps_voc/infer_aff.py \
#     --work_space ${WORKDIR} \
#     --checkpoint ${WORKDIR}/res38_aff_last.pth \
#     --infer_list configs/voc12/train_aug.txt \
#     --cam_out_dir cam_mask \
#     --seg_out_dir ${SEG_DIR} \


# #============= Evaluate =============#
# python steps_voc/eval_sem_seg.py \
#     --work_space ${WORKDIR} \
#     --seg_out_dir ${SEG_DIR} \

# Save the generated mask to zip
cd ${WORKDIR} && zip -r ${SEG_DIR}.zip ${SEG_DIR} && cd -