#!/bin/bash
GPU=0
NODES=1
DATASET=COCO
<<<<<<< HEAD
DATACONFIG=data/MSCOCO/ImageList
=======
DATACONFIG=data/MSCOCO/ImageLists

>>>>>>> 2f7ab046cf1e909ae270875a2e1eed2f335e3f5e
TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt

SEGDIR=cam_mask_val
CRFDIR=crf_mask_val
INPUTSIZE=448
MODELNAME=mcta
WORKDIR=results_coco/mcta

CUDA_VISIBLE_DEVICES=${GPU}

<<<<<<< HEAD
# # ============= Make Class Activation Maps of Model ============= #
# python make_cam.py \
#     --dataset ${DATASET} \
#     --model ${MODELNAME} \
#     --work_space ${WORKDIR} \
#     --train_list ${VALID} \
#     --input_size ${INPUTSIZE} \
#     --cam_out_dir ${SEGDIR} \
#     --checkpoint results_coco/mcta/mcta-deit-small-coco-4627.pth \
=======
# ============= Make Class Activation Maps of Model ============= #
python make_cam.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --work_space ${WORKDIR} \
    --train_list ${VALID} \
    --input_size ${INPUTSIZE} \
    --cam_out_dir ${SEGDIR} \
    --checkpoint results_coco/mcta/mcta-deit-small-coco-4627.pth \
>>>>>>> 2f7ab046cf1e909ae270875a2e1eed2f335e3f5e
      
# ============= Evaluate Class Activation Maps No CRF =============#
python eval_cam_crf.py \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --eval_cam_dir ${SEGDIR} \
    --crf_cam_dir ${CRFDIR} \
    --id_list ${VALID} \
    --curve_threshold \
    --low_thres 44 \
    --high_thres 52
    
# ============= Evaluate Class Activation Maps =============#
python eval_cam_crf.py \
    --use_crf \
    --dataset ${DATASET} \
    --work_space ${WORKDIR} \
    --eval_cam_dir ${SEGDIR} \
    --crf_cam_dir ${CRFDIR} \
<<<<<<< HEAD
    --alpha 1.2 \
    --id_list ${VALID} \
=======
    --id_list ${VALID} \

>>>>>>> 2f7ab046cf1e909ae270875a2e1eed2f335e3f5e

# #============= Evaluate =============#
# python steps_coco/eval_sem_seg.py \
#     --id_list ${VALID} \
#     --work_space ${WORKDIR} \
#     --seg_out_dir ${CRFDIR} \

# # Save the generated mask to zip
# cd ${WORKDIR} && zip -r ${CRFDIR}.zip ${CRFDIR} && cd -