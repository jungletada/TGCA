GPU=0
NODES=1
DATASET=COCO
DATACONFIG=data/MSCOCO/ImageLists

TRAINID=${DATACONFIG}/train_id.txt
VALID=${DATACONFIG}/val_id.txt
INPUTSIZE=448

MODELNAME=resnet38d
WORKDIR=results_coco/resnet38d


CUDA_VISIBLE_DEVICES=${GPU}

# ============= Train Model ===============#
OMP_NUM_THREADS=${NODES} \
torchrun --nproc_per_node=${NODES} --nnodes=1 \
    train_resnet.py \
    --dataset ${DATASET} \
    --model ${MODELNAME} \
    --train_list ${TRAINID} \
    --work_space ${WORKDIR} \
    --input_size ${INPUTSIZE} \
    --seed 3 \
    --epoch 35 \
    --batch_per_gpu 40 \