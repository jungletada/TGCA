import os
import sys
import logging
import datetime
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
import imageio.v2 as imageio

sys.path.append(osp.dirname(__file__) + os.sep + '../')
from dataloaders.coco.dataloader_psa import COCOSegmentationLabelDataset
from engine import calc_semantic_segmentation_confusion
import utils


def run(args):
    session_name = "pseudo_mask"
    
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M")
    utils.logger_info(logger_name=session_name,
                      log_path=osp.join(
                          args.work_space, f'{session_name}_{now}.log'))
    logger = logging.getLogger(session_name)
    
    dataset = COCOSegmentationLabelDataset(
        data_dir=args.mscoco_root,
        id_list_file=args.infer_list,
        annotation_dir='MaskSets')
    
    num_images = len(dataset)
    logger.info(f"COCO: Number of images = {num_images}")
    
    chunk_size = 10000     # for memory efficient
    split_indices = [(i, min(i + chunk_size, num_images))
                     for i in range(0, num_images, chunk_size)]
    
    def chunk_eval(begin_idx, end_idx):
        preds = []
        labels = []
        for i in tqdm(range(begin_idx, end_idx)):
            pack = dataset[i]
            img_name = pack['name_id']
            cls_file = img_name + '.png'
            cls_labels = imageio.imread(osp.join(args.seg_out_dir, cls_file)).astype(np.uint8)
            preds.append(cls_labels.copy())
            labels.append(pack['label'])
            
        confusion = calc_semantic_segmentation_confusion(preds, labels)
        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        chunk_miou = np.nanmean(gtjresj / denominator)
        return {'mIoU':chunk_miou, 'gtj':gtj, 'resj':resj, 'gtjresj': gtjresj}
    
    all_slices = {'mIoU': [], 'gtj': [], 'resj': [], 'gtjresj': []}
    for idx_pair in split_indices:
        result = chunk_eval(
            begin_idx=idx_pair[0], 
            end_idx=idx_pair[1])
        
        for key in all_slices.keys():
            all_slices[key].append(result[key])
    
    # mean_IoU = np.mean(all_slices['mIoU'])
    gtj = np.sum(all_slices['gtj'], axis=0)
    resj = np.sum(all_slices['resj'], axis=0)
    gtjresj = np.sum(all_slices['gtjresj'], axis=0)
    
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    mean_iou = np.nanmean(iou)

    logger.info("IoU(%) for each class:")
    for i, res in enumerate(iou):
        logger.info("  Class {:2d}, {:.2f}".format(i, res * 100.))
    logger.info("-- Mean IoU for all images: {:.2f}".format(mean_iou * 100.))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate Pseudo Masks', add_help=False)
    parser.add_argument("--work_space", default="results_coco/mcta", type=str)
    parser.add_argument("--mscoco_root", default='data/MSCOCO', type=str,help="Path to MSCOCO")
    parser.add_argument('--seg_out_dir', default='pseudo_mask', help='pseudo_mask directory')
    parser.add_argument('--infer_list', default='val_id.txt', help='list file path')
    args = parser.parse_args()
    args.seg_out_dir = osp.join(args.work_space, args.seg_out_dir)
    run(args=args)
