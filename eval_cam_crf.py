import os
import cv2

import torch
import argparse
import datetime
import numpy as np
import os.path as osp
from tqdm import tqdm
import PIL.Image as Image
from torch import multiprocessing
import torch.nn.functional as F

from misc import torchutils
from dataloaders.coco.dataloader_psa import COCOSegmentationLabelDataset
from dataloaders.voc12.dataloader_psa import VOCSegmentationLabelDataset
from engine import calc_semantic_segmentation_confusion
from misc.dcrf import DenseCRF


crf_inference = DenseCRF(
    iter_max=10,
    pos_xy_std=3,
    pos_w=3,
    bi_xy_std=45,
    bi_rgb_std=3,
    bi_w=3,
)


def logfile(args, msg):
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write(msg + '\n')
    print(msg)


def run_eval(dataset, args):
    num_images = len(dataset)
    chunk_size = 12000   # for memory efficient
    split_indices = [(i, min(i + chunk_size, num_images))
                     for i in range(0, num_images, chunk_size)]

    def eval(threshold, begin_idx, end_idx):
        preds = []
        labels = []
        miou = 0.
        for i in tqdm(range(begin_idx, end_idx)):
            pack = dataset[i]
            image = pack['image'] # H, W, 3
            filename = pack['name_id']
            cam_dict = np.load(osp.join(args.eval_cam_dir, filename + '.npy'),
                            allow_pickle=True).item()
            if len(tuple(cam_dict.keys())) == 0:
                continue
            cams = np.stack(list(cam_dict.values()), axis=0) # (#val_cls, H, W)
            prob = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            # bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), 1.4)
            # cams = np.concatenate((bg_score, cams), axis=0)
            # prob = crf_inference(image, cams, t=10, scale_factor=1, labels=cams.shape[0])
            
            cls_labels = np.argmax(prob, axis=0)
            keys = (torch.stack(tuple(cam_dict.keys())) + 1).numpy()
            keys = np.pad(keys, (1, 0), mode='constant')
            cls_labels = keys[cls_labels].astype(np.uint8)

            preds.append(cls_labels.copy())
            labels.append(pack['label'])

        confusion = calc_semantic_segmentation_confusion(preds, labels)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator
        miou = np.nanmean(iou)
        # print("Chunk for threshold: {:.2f} mIoU: {:.4f}".format(threshold, miou))
        # print('among_pred_fg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))
        return miou
    
    if args.curve_threshold:
        best_res = 0.
        best_threshold = 0
        for t in range(args.low_thres, args.high_thres):
            all_slices = []
            for idx_pair in split_indices:
                miou = eval(
                    t / 100., 
                    begin_idx=idx_pair[0], 
                    end_idx=idx_pair[1])
                all_slices.append(miou)
            mean_value = np.mean(all_slices)
            logfile(args, f"threshold={t / 100:.2f}; mIoU for all images: {mean_value * 100:.2f}%")
            if mean_value > best_res: 
                best_res = mean_value
                best_threshold = t / 100.
            else:
                break    
        logfile(args, 
                f"Best threshold: {best_threshold}, best mIoU: {best_res * 100:.2f}%, num_imgs: {num_images}")
        
    else:
        all_slices = []
        for idx_pair in split_indices:
            miou = eval(
                args.threshold,
                begin_idx=idx_pair[0],
                end_idx=idx_pair[1])
            all_slices.append(miou)
        mean_value = np.mean(all_slices)
        logfile(args, "mIoU for all images: {:.2f}%".format(mean_value * 100))


def make_cam_crf(process_id, dataset, args):
    """Generates Class Activation Maps (CAM) using CRF.

    Args:
        process_id (int): The index of the process.
        dataset (list): The dataset to evaluate.
        args (Namespace): The arguments containing configuration settings.
    """

    databin = dataset[process_id]
    num_images = len(databin)
    for i in tqdm(range(num_images)):
        pack = databin[i]
        image = pack['image']
        filename = pack['name_id']

        try:
            cam_dict = np.load(
                osp.join(args.eval_cam_dir, filename + '.npy'),
                allow_pickle=True).item()
        except EOFError as e:
            print(f'{e}, {filename}')
            
        if len(tuple(cam_dict.keys())) == 0:
            cam = np.zeros(image.shape[:2], np.uint8)
            mask = Image.fromarray(cam, mode='L')
            mask.save(osp.join(args.crf_cam_dir, filename + '.png'))
            continue
            
        cams = np.stack(list(cam_dict.values()), axis=0) # (#val_cls, H, W)
        bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), args.alpha)
        cams = np.concatenate((bg_score, cams), axis=0)
        prob = crf_inference(image, cams)
        
        cls_labels = np.argmax(prob, axis=0)
        keys = (torch.stack(tuple(cam_dict.keys())) + 1).numpy()
        keys = np.pad(keys, (1, 0), mode='constant')
        
        cls_labels = keys[cls_labels].astype(np.uint8)
        
        mask = Image.fromarray(cls_labels, mode='L')
        mask.save(osp.join(args.crf_cam_dir, filename + '.png'))
        
        # cams = np.stack(list(cam_dict.values()), axis=0) # (#val_cls, H, W)
        # bg_score_h = np.power(1 - np.max(cams, axis=0, keepdims=True), 1.5)
        # cams_h = np.concatenate((bg_score_h, cams), axis=0)
        # bg_score_l = np.power(1 - np.max(cams, axis=0, keepdims=True), 1)
        # cams_l = np.concatenate((bg_score_l, cams), axis=0)
        # prob_h = crf_inference(image, cams_h)
        # prob_l = crf_inference(image, cams_l)
        
        # keys = np.stack(tuple(cam_dict.keys())) + 1
        # keys = np.pad(keys, (1, 0), mode='constant')

        # pred_h = np.argmax(prob_h, axis=0)
        # pred_h = keys[pred_h]
        
        # pred_l = np.argmax(prob_l, axis=0)
        # pred_l = keys[pred_l]

        # pred = pred_h.copy()
        # pred[pred_h == 0] = 255
        # pred[(pred_h + pred_l) == 0] = 0

        # _pred = np.squeeze(pred).astype(np.uint8)
        # mask = Image.fromarray(_pred, mode='L')
        # mask.save(osp.join(args.crf_cam_dir, filename + '.png'))

     
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate CAMs', add_help=False)
    parser.add_argument('--use_crf', action='store_true', help='use crf to make CAM.')
    parser.add_argument('--dataset', default='', type=str, help='name of dataset')
    parser.add_argument('--mscoco_root', default='data/MSCOCO', type=str, help='COCO dataset path')
    parser.add_argument('--voc12_root', default='data/VOCdevkit/VOC2012', type=str, help='VOC12 dataset path')
    parser.add_argument("--id_list", default="train_aug_id.txt", type=str,
                        help='train_id.txt or train_aug_id.txt')
    parser.add_argument('--work_space', default='results', help='work space directory')
    parser.add_argument('--eval_cam_dir', default='cam_mask', help='cam_mask directory')
    parser.add_argument('--log_file', default='eval_cam.log', type=str,
                        help='log file to save the results')
    parser.add_argument('--log_dir', default='log_dir', type=str,
                        help='log dir to save the results')
    parser.add_argument('--curve_threshold', action='store_true', help='whether to use a range of thresholds')
    parser.add_argument('--threshold', default=0.45, type=float, help='threshold for evaluation as background')
    parser.add_argument('--low_thres', default=42, type=int, help='low threshold for evaluation as background')
    parser.add_argument('--high_thres', default=55, type=int, help='high threshold for evaluation as background')
    parser.add_argument('--alpha', default=1.15, type=float, help='use alpha to set background')
    parser.add_argument('--eval_nprocs', default=8, type=int, help='use nprocs processess.')
    parser.add_argument("--crf_cam_dir", default="crf_mask", type=str, help="crf mask path")
    args = parser.parse_args()
    #----------------------------------------------------------------------------------#
    args.eval_cam_dir = osp.join(args.work_space, args.eval_cam_dir)
    args.crf_cam_dir = osp.join(args.work_space, args.crf_cam_dir)
    args.log_dir = osp.join(args.work_space, args.log_dir)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.crf_cam_dir, exist_ok=True)

    if args.dataset == 'VOC12':
        dataset = VOCSegmentationLabelDataset(
            data_dir=args.voc12_root,
            id_list_file=args.id_list)
        # args.low_thres, args.high_thres = 44, 55

    elif args.dataset == 'COCO':
        dataset = COCOSegmentationLabelDataset(
            data_dir=args.mscoco_root,
            id_list_file=args.id_list,
            annotation_dir='MaskSets')
        # args.low_thres, args.high_thres = 44, 55
    else:
        raise NotImplementedError

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    current_session = 'train' if 'train' in args.id_list else 'val'
    if not args.use_crf:
        args.log_file = osp.join(args.log_dir,
                                 f"eval-cam-crf-{current_session}-{time}.log")
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write(f"{time}: Evaluating CAMs for {args.dataset}\n")
        run_eval(args=args, dataset=dataset)
        torch.cuda.empty_cache()
        
    else:
        args.log_file = osp.join(args.log_dir,
                                 f"eval-cam-{current_session}-{time}.log")
        EVAL_NPROCS = args.eval_nprocs
        split_dataset = torchutils.split_dataset(dataset, EVAL_NPROCS)
        multiprocessing.spawn(
            make_cam_crf,
            nprocs=EVAL_NPROCS,
            args=(split_dataset, args),
            join=True)
        torch.cuda.empty_cache()