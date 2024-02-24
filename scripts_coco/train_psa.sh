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


# #============= Train Pixel Semantic Affnity =============#
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     steps_coco/train_aff.py \
#     --work_space ${WORKDIR} \
#     --model_weights checkpoints/res38_cls.pth \
#     --batch_per_gpu 4 \
#     --seed 3 \
#     --low_alpha 1 \
#     --high_alpha 3 \
    

# #============= Infer with Pixel Semantic Affnity =============#
# python steps_coco/infer_aff.py \
#     --work_space ${WORKDIR} \
#     --checkpoint ${WORKDIR}/res38_aff_last.pth \
#     --cam_out_dir cam_mask \
#     --seg_out_dir pseudo_mask \
#     --threshold 0.41 \


#============= Evaluate =============#
python steps_coco/eval_sem_seg.py \
    --work_space ${WORKDIR} \
    --seg_out_dir pseudo_mask \
   