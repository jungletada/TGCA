#!/bin/bash
echo "  Available Dataset: VOC12, COCO";
echo "  |-- VOC12 dataset path: datasets/VOCdevkit/VOC2012";
echo "  |-- COCO dataset path: datasets/MSCOCO";

# NEED TO SET
GPU=0,1
NODES=2
MODEL=mctgformer
CUDA_VISIBLE_DEVICES=${GPU}
OMP_NUM_THREADS=${NODES} 
WORKDIR=results_voc/${MODEL}


# ============= Train Model ===============#
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    train_model.py \
    --model deit_small_mctgformer \
    --work_space ${WORKDIR} \
    --seed 2 \
    --epoch 45 \
    --batch_per_gpu 16 \
    --data_set VOC12 \


# ============= Make Class Activation Maps of Model=============#
python steps_voc/make_cam.py \
    --model deit_small_${MODEL} \
    --work_space ${WORKDIR} \
    --checkpoint ${WORKDIR}/deit_small_${MODEL}_best.pth \
    --infer_list configs/voc12/train_id.txt \


# ============= Evaluate Class Activation Maps =============#
python steps_voc/eval_cam.py \
    --curve_threshold \
    --work_space ${WORKDIR} \


# # #============= Evaluate Class Activation Maps =============#
# python evaluation.py \
#     --work_space ${WORKDIR} \
#     --infer_list configs/voc12/train_id.txt \
#     --log_file eval_seeds.log \
#     --predict_dir ${WORKDIR}/cam_mask \
#     --type npy \
#     --curve True \

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

# # Save the generated mask to zip
# cd ${WORKDIR} && zip -r ${SEG_DIR}.zip ${SEG_DIR} && cd -


# OMP_NUM_THREADS=${NODES}  \
#     torchrun \
#     --nproc_per_node=${NODES} --nnodes=1  \
#     steps_voc/train_eval_seg.py \
#     --train True \
#     --seed 1 \
#     --num_epochs 30 \
#     --batch_per_gpu 8 \
#     --init_weights checkpoints/res38_cls.pth \


# python steps_voc/train_eval_seg.py \
#     --evaluate True \
#     --use_crf True \
#     --scales 1.0 \
#     # --pred_path val_ms \