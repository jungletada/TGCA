import torch
import datetime
import argparse
import numpy as np
import os.path as osp
from engine import calc_semantic_segmentation_confusion


def logfile(args, msg):
    with open(args.log_file, "a") as f:
        f.write(msg + '\n')
    print(msg)
    
    
def run_eval_cam(args, dataset):
    num_images = len(dataset)
    def eval_curve(threshold):
        preds = []
        labels = []
        miou = 0.
        for i, img_id in enumerate(dataset.ids):
            cam_dict = np.load(osp.join(args.eval_cam_dir, img_id + '.npy'), allow_pickle=True).item()
            
            cams = np.stack(list(cam_dict.values()), axis=0) # (#val_cls, H, W)
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
            
            keys = (torch.stack(tuple(cam_dict.keys())) + 1).numpy()
            keys = np.pad(keys, (1, 0), mode='constant')
            
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            
            preds.append(cls_labels.copy())
            labels.append(dataset[i]["label"])

        confusion = calc_semantic_segmentation_confusion(preds, labels)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator
        miou = np.nanmean(iou)
        logfile(args, "threshold: {:.2f}, miou: {:.4f}".format(threshold, miou))
        # print('among_pred_fg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))
        return miou
    
    if args.curve_threshold:
        best_res = 0.
        best_threshold = 0
        
        for t in range(35, 55):
            miou = eval_curve(t / 100.)
            if miou > best_res: 
                best_res = miou
                best_threshold = t / 100.
            else:
                break
        time = datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")      
        logfile(args, "{} Best threshold: {}, best miou: {:.4f}, num_imgs: {}\n".format(
            time, best_threshold, best_res, num_images))
        
    else:
        miou = eval_curve(args.threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate CAMs', add_help=False)
    parser.add_argument('--voc12_root', default='datasets/VOCdevkit/VOC2012', type=str, help='VOC12 dataset path')
    parser.add_argument("--train_list", default="configs/voc12/train_aug_id.txt", type=str, 
                        help='configs/coco/train_id.txt or configs/voc12/train_aug_id.txt')
    
    parser.add_argument('--work_space', default='results_voc/your_model', help='work space directory')
    parser.add_argument('--eval_cam_dir', default='cam_mask', help='cam_mask directory')
    
    parser.add_argument('--curve_threshold', action='store_true', help='whether to use a range of thresholds')
    parser.add_argument('--threshold', default=0.43, type=float, help='threshold for evaluation as background')
    args = parser.parse_args()
    #---------------------------------------------------------------#
    args.eval_cam_dir = osp.join(args.work_space, args.eval_cam_dir)
    
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M") 
    args.log_file = osp.join(args.work_space, f"eval_cam_{time}.log")
    with open(args.log_file, "a") as f:
        f.write("{}: Evaluation class activation map for VOC2012\n".format(time))
        
    from data.voc12.dataloader_psa import VOCSegmentationLabelDataset
    dataset = VOCSegmentationLabelDataset( 
        data_dir=args.voc12_root, 
        id_list_file=args.train_list)
    
    run_eval_cam(args=args, dataset=dataset)
    
    