# NEED TO SET
GPU=0,1
WORKDIR=results_voc/MCTG
CUDA_VISIBLE_DEVICES=${GPU} \


# ================= Use pseudo mask for segmentation training =================#
OMP_NUM_THREADS=2  \
    torchrun \
    --nproc_per_node=2 --nnodes=1  \
    seg/train_seg.py \
    --num_epochs 30 \
    --num_classes 21 \
    --batch_per_gpu 8 \
    --network resnet38_seg \
    --save_path ${WORKDIR} \
    --init_weights checkpoints/res38_cls.pth \
    --list_path configs/voc12/train_aug_id.txt \
    --seg_pgt_path datasets/VOCdevkit/VOC2012/SegmentationClassAug \
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