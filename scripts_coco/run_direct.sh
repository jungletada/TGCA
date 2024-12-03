#!/bin/bash
GPU=0,1
NODES=2
DATASET=COCO
DATACONFIG=data/MSCOCO/ImageList

TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt

SEGDIR=cam_mask_val
CRFDIR=crf_mask_val

INPUTSIZE=448

MODELNAME=mcta
WORKDIR=results_coco/mcta

CUDA_VISIBLE_DEVICES=${GPU}

# # ============= Make Class Activation Maps of Model ============= #
# python make_cam.py \
#     --dataset ${DATASET} \
#     --model ${MODELNAME} \
#     --work_space ${WORKDIR} \
#     --train_list ${VALID} \
#     --input_size ${INPUTSIZE} \
#     --cam_out_dir ${SEGDIR} \
#     --checkpoint results_coco/mcta/mcta-deit-small-coco-4572.pth \
      
# # ============= Evaluate Class Activation Maps No CRF =============#
# python eval_cam_crf.py \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --eval_cam_dir ${SEGDIR} \
#     --crf_cam_dir ${CRFDIR} \
#     --id_list ${VALID} \
#     --curve_threshold \
#     --low_thres 44 \
#     --high_thres 50
    
# # ============= Evaluate Class Activation Maps =============#
# python eval_cam_crf.py \
#     --use_crf \
#     --dataset ${DATASET} \
#     --work_space ${WORKDIR} \
#     --eval_cam_dir ${SEGDIR} \
#     --crf_cam_dir ${CRFDIR} \
#     --id_list ${VALID} \


#============= Evaluate =============#
python steps_coco/eval_sem_seg.py \
    --id_list ${VALID} \
    --work_space ${WORKDIR} \
    --seg_out_dir ${CRFDIR} \

# # Save the generated mask to zip
# cd ${WORKDIR} && zip -r ${CRFDIR}.zip ${CRFDIR} && cd -