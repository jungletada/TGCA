import torch
import argparse
import datetime
import numpy as np
import os.path as osp
from tqdm import tqdm

from data.coco.dataloader_psa import COCOSegmentationLabelDataset
from data.voc12.dataloader_psa import VOCSegmentationLabelDataset
from engine import calc_semantic_segmentation_confusion


def logfile(args, msg):
    with open(args.log_file, "a") as f:
        f.write(msg + '\n')
    print(msg)
    
    
def run(args, dataset):
    num_images = len(dataset)
    chunk_size = 12000   # for memory efficient
    split_indices = [(i, min(i + chunk_size, num_images)) 
                     for i in range(0, num_images, chunk_size)]
    
    def eval_curve(threshold, begin_idx, end_idx):
        preds = []
        labels = []
        miou = 0.
        for i in tqdm(range(begin_idx, end_idx)):
            pack = dataset[i]
            filename = pack['name_id']
            try:
                cam_dict = np.load(
                    osp.join(args.eval_cam_dir, filename + '.npy'), 
                    allow_pickle=True).item()
            except EOFError as e:
                print(f'{e}, {filename}')
                
            cams = np.pad(cam_dict['high_res'], ((1, 0), (0, 0), (0, 0)), 
                          mode='constant', constant_values=threshold)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant') # [0, cls1, ...]      
            
            cls_labels = np.argmax(cams, axis=0)
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
        # time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        # print("{} Chunk for threshold: {:.2f} mIoU: {:.4f}".format(time, threshold, miou))
        # print('among_pred_fg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))
        return miou
    
    if args.curve_threshold:
        best_res = 0.
        best_threshold = 0
        for t in range(args.low_thres, args.high_thres):
            all_slices = []
            for idx_pair in split_indices:
                miou = eval_curve(
                    t / 100., 
                    begin_idx=idx_pair[0], 
                    end_idx=idx_pair[1])
                all_slices.append(miou)
            mean_value = np.mean(all_slices)
            logfile(args, "threshold={:.2f}; mean IoU for all images: {:.2f}%".format(
                t / 100., mean_value * 100.))
            if mean_value > best_res: 
                best_res = mean_value
                best_threshold = t / 100.
            else:
                break    
        logfile(args, "Best threshold: {}, best miou: {:.2f}%, num_imgs: {}".format(
            best_threshold, best_res * 100., num_images))
        
    else:
        all_slices = []
        for idx_pair in split_indices:
            miou = eval_curve(
                args.threshold, 
                begin_idx=idx_pair[0], 
                end_idx=idx_pair[1])
            all_slices.append(miou)
        mean_value = np.mean(all_slices)
        logfile(args, "mean IoU for all images: {:.2f}%".format(mean_value * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate CAMs', add_help=False)
    parser.add_argument('--dataset', default='', type=str, help='name of dataset')
    parser.add_argument('--mscoco_root', default='datasets/MSCOCO', type=str, help='COCO dataset path')
    parser.add_argument('--voc12_root', default='datasets/VOCdevkit/VOC2012', type=str, help='VOC12 dataset path')
    parser.add_argument("--train_list", default="configs/voc12/train_aug_id.txt", type=str, 
                        help='configs/coco/train_id.txt or configs/voc12/train_aug_id.txt')
    parser.add_argument('--work_space', default='results_coco/MCTG', help='work space directory')
    parser.add_argument('--eval_cam_dir', default='cam_mask', help='cam_mask directory')
    parser.add_argument('--log_file', default='eval_cam.log', type=str, 
                        help='log file to save the results')
    parser.add_argument('--curve_threshold', action='store_true', help='whether to use a range of thresholds')
    parser.add_argument('--threshold', default=0.45, type=float, help='threshold for evaluation as background')
    args = parser.parse_args()
    #----------------------------------------------------------------------------------#
    args.eval_cam_dir = osp.join(args.work_space, args.eval_cam_dir)
    args.log_file = osp.join(args.work_space, args.log_file)

    if args.dataset == 'VOC12':
        dataset = VOCSegmentationLabelDataset( 
            data_dir=args.voc12_root, 
            id_list_file=args.train_list)
        args.low_thres, args.high_thres = 40, 55

    elif args.dataset == 'COCO':
        dataset = COCOSegmentationLabelDataset(
            data_dir=args.mscoco_root, 
            id_list_file=args.train_list,
            annotation_dir='MaskSets')
        args.low_thres, args.high_thres = 43, 50

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") 
    args.log_file = osp.join(args.work_space, f"eval_cam_{time}.log")
    with open(args.log_file, "a") as f:
        f.write("{}: Evaluation class activation map for {}\n".format(time, args.dataset))
        
    run(args=args, dataset=dataset)