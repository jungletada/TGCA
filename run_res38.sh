# NEED TO SET
GPU=0,1
NODES=2
DATASET=VOC12
MODEL=resnet38d
INPUTSIZE=224
CUDA_VISIBLE_DEVICES=${GPU}
WORKDIR=results_voc/${MODEL}

DATACONFIG=configs/voc12
TRAINAUGID=${DATACONFIG}/train_aug_id.txt
TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt
TRAINAUGLIST=${DATACONFIG}/train_aug.txt
TRAINLIST=${DATACONFIG}/train.txt


# ==================== Train ResNet38 ==================== #
OMP_NUM_THREADS=${NODES} \
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    train_res38.py \
    --dataset ${DATASET} \
    --model ResNet38d_patch_224 \
    --work_space ${WORKDIR} \
    --input_size ${INPUTSIZE} \
    --seed 8 \
    --epoch 10 \
    --batch_per_gpu 16 \