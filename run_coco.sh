#!/bin/bash
echo "  Available Dataset: VOC12, COCO";
echo "  |-- VOC12 dataset path: datasets/VOCdevkit/VOC2012";
echo "  |-- COCO dataset path: datasets/MSCOCO";


GPU=0,1
NODES=2
DATASET=COCO
DATACONFIG=configs/coco

TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt

INPUTSIZE=448
SEG_DIR=pseudo_mask

MODELNAME=msgformer
WORKDIR=results_coco/msgformer_${INPUTSIZE}


CUDA_VISIBLE_DEVICES=${GPU}

# # ============= Train Model ===============#
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_model.py \
#     --dataset ${DATASET} \
#     --model ${MODELNAME} \
#     --train_list ${TRAINID} \
#     --work_space ${WORKDIR} \
#     --input_size ${INPUTSIZE} \
#     --seed 3 \
#     --epoch 29 \
#     --batch_per_gpu 19 \
    

# ============= Make Class Activation Maps of Model=============#
python make_cam.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --work_space ${WORKDIR} \
    --train_list ${TRAINID} \
    --input_size ${INPUTSIZE} \
    --checkpoint ${WORKDIR}/msgformer_4550.pth \
    # --checkpoint ${WORKDIR}/${MODELNAME}_best.pth \
    

# ============= Evaluate Class Activation Maps =============#
python eval_cam_crf.py \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --train_list ${TRAINID} \
    --curve_threshold \


# #============= Train and Infer Pixel Semantic Affnity =============
# OMP_NUM_THREADS=${NODES} \
# torchrun --nproc_per_node=${NODES} --nnodes=1 \
#     train_infer_psa.py --train True \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --train_list ${TRAINID} \
#     --weights checkpoints/res38_cls.pth \
#     --seed 3 \
#     --epoch 5 \
#     --low_alpha 1 \
#     --high_alpha 1.2 \


# #============= Infer with Pixel Semantic Affnity =============#
# python train_infer_psa.py --inference True \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --infer_list ${TRAINID} \
#     --seg_out_dir ${SEG_DIR} \
#     --threshold 0.45 \


# #============= Evaluate =============#
# python steps_coco/eval_sem_seg.py \
#     --work_space ${WORKDIR} \
#     --seg_out_dir ${SEG_DIR} \
#     # --seg_out_dir pseudo_mask_448 \

# # Save the generated mask to zip
# cd ${WORKDIR} && zip -r ${SEG_DIR}.zip ${SEG_DIR} && cd -