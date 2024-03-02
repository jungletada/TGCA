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


# ============= Train Model ===============#
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    train_res38.py \
    --model ResNet38d_patch_224 \
    --work_space results_voc/resnet38 \
    --seed 2 \
    --epoch 45 \
    --batch_per_gpu 16 \
    --data_set VOC12 \

# ============= Train Model ===============#
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    train_v2.py \
    --model deit_small_MCTG \
    --seed 2 \
    --epoch 45 \
    --batch_per_gpu 16 \
    --work_space ${WORKDIR} \
    --data_set VOC12 \


# ============= Make Class Activation Maps of Model=============#
python steps_voc/make_cam.py \
    --model deit_small_${MODEL} \
    --checkpoint ${WORKDIR}/deit_small_${MODEL}_best.pth \
    --infer_list configs/voc12/train_id.txt \


#============= Evaluate Class Activation Maps =============#
python evaluation.py \
    --work_space ${WORKDIR} \
    --infer_list configs/voc12/train_id.txt \
    --log_file eval_seeds.log \
    --predict_dir ${WORKDIR}/cam_mask \
    --type npy \
    --curve True \


#============= Train Pixel Semantic Affnity =============#
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    steps_voc/train_aff.py \
    --work_space ${WORKDIR} \
    --batch_per_gpu 4 \
    --seed 2 \
    --low_alpha 1 \
    --high_alpha 3 \
    --model_weights checkpoints/res38_cls.pth \


#============= Infer with Pixel Semantic Affnity =============#
python steps_voc/infer_aff.py \
    --work_space ${WORKDIR} \
    --checkpoint ${WORKDIR}/res38_aff_last.pth \
    --infer_list configs/voc12/train_aug.txt \
    --cam_out_dir cam_mask \
    --seg_out_dir ${SEG_DIR} \


#============= Evaluate =============#
python steps_voc/eval_sem_seg.py \
    --work_space ${WORKDIR} \
    --seg_out_dir ${SEG_DIR} \

# Save the generated mask to zip
cd ${WORKDIR} && zip -r ${SEG_DIR}.zip ${SEG_DIR} && cd -


#================= Use pseudo mask for segmentation training =================#
OMP_NUM_THREADS=2  \
    torchrun \
    --nproc_per_node=2 --nnodes=1  \
    seg/train_seg.py \
    --seed 4 \
    --num_epochs 30 \
    --num_classes 21 \
    --batch_per_gpu 8 \
    --network resnet38_seg \
    --save_path ${WORKDIR} \
    --seg_pgt_path ${WORKDIR}/pseudo_mask \
    --init_weights results_voc/resnet38/ResNet38d_patch_224_best.pth \
    --list_path configs/voc12/train_aug_id.txt \
    --img_path datasets/VOCdevkit/VOC2012/JPEGImages \
   

#================= Evaluate the segmentation results =================#
python seg/infer_seg.py \
    --dataset voc \
    --save_palette False \
    --use_crf True \
    --scales 0.5 0.75 1.0 1.25 1.5 \
    --weights ${WORKDIR}/resnet38_seg_last.pth \
    --list_path configs/voc12/val_id.txt \
    --gt_path datasets/VOCdevkit/VOC2012/SegmentationClass \
    --img_path datasets/VOCdevkit/VOC2012/JPEGImages \
    # --save_path ${WORKDIR}/val_ms_crf \