import os
import sys
import imageio
import logging
import argparse
import numpy as np
import os.path as osp
from torch.utils.data import DataLoader
from tqdm import tqdm
from chainercv.evaluations import calc_semantic_segmentation_confusion

sys.path.append(osp.dirname(__file__) + os.sep + '../')
from data.coco.dataloader import COCOSegmentationDataset
import utils


def run(args):
    session_name = "pseudo_mask"
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M")
    utils.logger_info(logger_name=session_name, 
                      log_path=osp.join(
                          args.work_space, f'{session_name}_{now}.log'))
    logger = logging.getLogger(session_name)
    
    dataset = COCOSegmentationDataset(
        image_dir = osp.join(args.coco_root,'train2014/'),
        anno_path= osp.join(args.coco_root,'annotations/instances_train2014.json'),
        masks_path=osp.join(args.coco_root,'mask/train2014'),
        crop_size=512)
    preds = []
    labels = []
    num_imgs = len(dataset)
    logger.info(f"COCO: Number of images = {num_imgs}")
    
    for i, pack in enumerate(tqdm(dataset)):
        img_name = pack['name'].split('.')[0]
        cls_file = img_name + '.png'
        cls_labels = imageio.imread(osp.join(args.seg_out_dir, cls_file)).astype(np.uint8)
        preds.append(cls_labels.copy())
        label = dataset.get_label_by_name(img_name)
        labels.append(label)
        
    confusion = calc_semantic_segmentation_confusion(preds, labels)
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator
    # print("Total images=", n_img)
    logger.info("False Positive: {:.4f}, False Negative: {:.4f}".format(fp[0], fn[0]))
    # print("IoU(%) for each class:")
    # for res in iou:
    #     print("{:.2f}".format(res * 100.))
    logger.info("mIoU (%): {:.2f}".format(np.nanmean(iou) * 100.))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate Pseudo Masks', add_help=False)
    parser.add_argument("--work_space", default="results_coco/MCTG", type=str)
    parser.add_argument("--coco_root", default='datasets/MSCOCO', type=str,help="Path to MSCOCO")
    parser.add_argument('--seg_out_dir', default='pseudo_mask', help='pseudo_mask directory')
    args = parser.parse_args()
    args.seg_out_dir = osp.join(args.work_space, args.seg_out_dir)
    run(args=args)
