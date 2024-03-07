import os
import sys
import torch
import argparse
import datetime
import numpy as np
import os.path as osp
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from data.coco.dataloader_psa import COCOSegmentationLabelDataset
from engine import calc_semantic_segmentation_confusion


def write_msg(file, msg):
    print(msg)
    file.write(msg+'\n')
    
    
def run(args):
    f = open(args.log_file, "w")
    dataset = COCOSegmentationLabelDataset(
        data_dir=args.mscoco_root, 
        id_list_file="configs/coco/train_id.txt",
        annotation_dir='MaskSets')
    
    num_images = len(dataset)
    chunk_size = 10000 # for memory efficient
    split_indices = [(i, min(i + chunk_size, num_images)) 
                     for i in range(0, num_images, chunk_size)]
    
    def eval_curve(threshold, begin_idx, end_idx):
        preds = []
        labels = []
        miou = 0.
        for i in tqdm(range(begin_idx, end_idx)):
            pack = dataset[i]
            filename = pack['name_id']
            cam_dict = np.load(osp.join(args.eval_cam_dir, filename + '.npy'), 
                               allow_pickle=True).item()
            
            if len(tuple(cam_dict.keys())) == 0:
                continue
            
            cams = np.stack(list(cam_dict.values()), axis=0) # (#val_cls, H, W)
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            
            keys = (torch.stack(tuple(cam_dict.keys())) + 1).numpy()
            keys = np.pad(keys, (1, 0), mode='constant')
            
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels].astype(np.uint8)
            
            preds.append(cls_labels.copy())
            labels.append(pack['label'])
            # if  i % 100 == 0:
            #     print("[{:5d}]".format(i))

        confusion = calc_semantic_segmentation_confusion(preds, labels)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator
        miou = np.nanmean(iou)
        
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
        # print("{} Chunk for threshold: {:.2f} mIoU: {:.4f}".format(time, threshold, miou))
        # print('among_pred_fg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))
        return miou
    
    if args.curve_threshold:
        best_res = 0.
        best_threshold = 0
        
        for t in range(40, 50):
            miou = eval_curve(t / 100.)
            if miou > best_res: 
                best_res = miou
                best_threshold = t / 100.
            else:
                break
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")      
        print("{} Best threshold: {}, best miou: {:.4f}, num_imgs: {}".format(
            time, best_threshold, best_res, num_images))
        
    
    else:
        all_slices = []
        for idx_pair in split_indices:
            miou = eval_curve(
                args.threshold, 
                begin_idx=idx_pair[0], 
                end_idx=idx_pair[1])
            all_slices.append(miou)
        mean_value = np.mean(all_slices)
        print("mean IoU for all images: {:.2f}".format(mean_value))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate CAMs', add_help=False)
    parser.add_argument('--mscoco_root', default='datasets/MSCOCO', type=str, help='COCO dataset path')
    parser.add_argument('--work_space', default='results_coco/MCTG', help='work space directory')
    parser.add_argument('--eval_cam_dir', default='cam_mask', help='cam_mask directory')
    parser.add_argument('--log_file', default='eval_cam.log', type=str, 
                        help='log file to save the results')
    parser.add_argument('--curve_threshold', action='store_true', help='whether to use a range of thresholds')
    parser.add_argument('--threshold', default=0.42, type=float, help='threshold for evaluation as background')
    args = parser.parse_args()
    #---------------------------------------------------------------#
    args.eval_cam_dir = osp.join(args.work_space, args.eval_cam_dir)
    args.log_file = osp.join(args.work_space, args.log_file)
    
    run(args=args)