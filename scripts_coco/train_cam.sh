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
WORKDIR=results_coco/${MODEL}


# # ============= Train Model ===============#
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=2 --nnodes=1 \
#     train_model.py \
#     --work_space ${WORKDIR} \
#     --model deit_small_${MODEL} \
#     --batch_per_gpu 16 \
#     --data_set COCO \
#     --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth


# # ============= Make Class Activation Maps of Model=============#
# python steps_coco/make_cam_coco.py \
#     --model deit_small_${MODEL} \
#     --checkpoint ${WORKDIR}/deit_small_${MODEL}_best.pth \ 


#============= Evaluate Class Activation Maps =============#
python steps_coco/eval_cam.py \
    --work_space ${WORKDIR} \