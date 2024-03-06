#!/bin/bash
echo "  Available Dataset: VOC12, COCO";
echo "  |-- VOC12 dataset path: datasets/VOCdevkit/VOC2012";
echo "  |-- COCO dataset path: datasets/MSCOCO";

# NEED TO SET
GPU=0,1
NODES=2
MODEL=MCTG
CUDA_VISIBLE_DEVICES=${GPU}
OMP_NUM_THREADS=${NODES} 
WORKDIR=results_voc/${MODEL}

# # ============= Train Model ===============#
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_res38.py \
#     --model ResNet38d_patch_224 \
#     --work_space results_voc/resnet38 \
#     --seed 2 \
#     --epoch 45 \
#     --batch_per_gpu 16 \
#     --data_set VOC12 \

# # ============= Train Model ===============#
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_v2.py \
#     --model deit_small_MCTG \
#     --seed 2 \
#     --epoch 45 \
#     --batch_per_gpu 16 \
#     --work_space ${WORKDIR} \
#     --data_set VOC12 \


# # ============= Make Class Activation Maps of Model=============#
# python steps_voc/make_cam.py \
#     --model deit_small_${MODEL} \
#     --checkpoint ${WORKDIR}/deit_small_${MODEL}_best.pth \
#     --infer_list configs/voc12/train_id.txt \


# #============= Evaluate Class Activation Maps =============#
# python evaluation.py \
#     --work_space ${WORKDIR} \
#     --infer_list configs/voc12/train_id.txt \
#     --log_file eval_seeds.log \
#     --predict_dir ${WORKDIR}/cam_mask \
#     --type npy \
#     --curve True \