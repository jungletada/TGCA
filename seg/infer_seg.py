import cv2
import sys,os
import os.path
import importlib
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.backends import cudnn
import torch.nn.functional as F

from tool import imutils
from tool.metrics import Evaluator
import PIL.Image as Image

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import logging
import utils 
cudnn.enabled = True
# mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
mean = np.expand_dims(np.array([0.485, 0.456, 0.406]), axis=(0, 1))
std = np.expand_dims(np.array([0.229, 0.224, 0.225]), axis=(0, 1))

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]


classes = np.array(('background',  # always index 0
                    'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse',
                    'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor'))


def crf_postprocess(pred_prob, ori_img, labels=21):
    crf_score = imutils.crf_inference_inf(ori_img, pred_prob, labels=labels)
    return crf_score


def str2bool(v):
    if v.lower() in ('yes','true','t','y','1','True'):
        return True
    elif v.lower() in ('no','false','f','n','0','False'):
        return False


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='voc')
    parser.add_argument('--model', type=str, default='MCTformerV2')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument("--weights", default="", type=str)
    parser.add_argument("--network", default="", type=str)
    parser.add_argument("--gt_path", required=True, type=str)
    # parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--save_palette", default=True, type=str2bool)
    parser.add_argument("--list_path", default="configs/voc12/val_id.txt", type=str)
    parser.add_argument("--img_path", default="", type=str)
    parser.add_argument("--num_classes", default=21, type=int)
    parser.add_argument("--use_crf", default=True, type=str2bool)
    parser.add_argument("--scales", type=float, nargs='+')
    args = parser.parse_args()
    return args


def transform_image(img_temp):
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float64)
    img_original = img_temp.astype(np.uint8)
    img_temp = (img_temp / 255.0 - mean) / std
    img_tensor = torch.from_numpy(img_temp[np.newaxis, :].transpose(0, 3, 1, 2)).float().cuda()
    return img_original, img_tensor
    
    
if __name__ == '__main__':
    args = get_args_parser()
    
    model_dir = utils.get_model_name(args.model)
    dataset_dir = utils.get_dataset_dir(args.dataset)
    
    base_save_path = f'results/{model_dir}/{dataset_dir}'
    utils.data_mkdir(base_save_path)
    
    session_name = "Segmentation Inference"
    log_path = os.path.join(base_save_path, 'eval_segmentation.log')
    utils.logger_info(logger_name=session_name, log_path=log_path)
    logger = logging.getLogger(session_name)
    logger.info(f"Model: {model_dir}, Dataset: {dataset_dir}")
    logger.info(f"Evaluation log path: {log_path}")
    logger.info(f"Multi-scale test: {tuple(args.scales)}, use CRF: {args.use_crf}")
   
    if args.save_palette: 
        pred_save_path = os.path.join(base_save_path, 'val_ms_crf_color')
        utils.data_mkdir(pred_save_path)
        logger.info(f"Multi-scale Evaluation with CRF save path: {pred_save_path}")
    else:
        logger.info("No to save Multi-scale Evaluation results.")
        
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    model = getattr(importlib.import_module('network.' + args.network), 'Net')(num_classes=args.num_classes)

    model.load_state_dict(torch.load(args.weights))
    seg_evaluator = Evaluator(num_class=args.num_classes)
    
    model.eval()
    model.cuda()
    base_img_path = args.img_path
    name_list = open(utils.get_dataset_imglist(args.dataset)).readlines()
    
    with torch.no_grad():
        for idx in tqdm(range(len(name_list))):
            i = name_list[idx]
            img_temp = cv2.imread(os.path.join(base_img_path, i.strip() + '.jpg'))
            img_original, img_tensor = transform_image(img_temp=img_temp)
            N, C, H, W = img_tensor.size()
            probs = torch.zeros((N, args.num_classes, H, W)).cuda()
            
            if args.scales:
                scales = tuple(args.scales)
                
            for s in scales:
                new_hw = [int(H * s), int(W * s)]
                img_scale = F.interpolate(img_tensor, new_hw, mode='bilinear', align_corners=True)
                prob = model(x=img_scale)
                prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=False)
                prob = F.softmax(prob, dim=1)
                probs = torch.max(probs, prob)

            output = probs.cpu().data[0].numpy()

            if args.use_crf:
                crf_output = crf_postprocess(output, img_original)
                pred = np.argmax(crf_output, 0)
            else:
                pred = np.argmax(output, axis=0)

            gt = Image.open(os.path.join(args.gt_path, i.strip() + '.png'))
            gt = np.asarray(gt)
            seg_evaluator.add_batch(gt, pred)

            # save_path = os.path.join(args.save_path, i.strip() + '.png')
            # cv2.imwrite(save_path, pred.astype(np.uint8))

            if args.save_palette:
                out = pred.astype(np.uint8)
                out = Image.fromarray(out, mode='P')
                out.putpalette(palette)
                out_name = os.path.join(pred_save_path, i.strip() + '.png')
                out.save(out_name)

        IoU, mIoU = seg_evaluator.Mean_Intersection_over_Union()

        str_format = "{:<15s}\t{:<15.2%}"
        for k in range(args.num_classes):
            logger.info('class {:2d} {:12} IU {:.3f}'.format(k, classes[k], IoU[k]))
        logger.info('mIoU = {:.3f}'.format(mIoU))
        
    #     # filename = os.path.join(base_save_path, 'eval_segmentation.log')
    #     # with open(filename, 'w') as f:
    #         for k in range(args.num_classes):
    #             # print(str_format.format(classes[k], IoU[k]))
    #             logger.info('class {:2d} {:12} IU {:.3f}'.format(k, classes[k], IoU[k]))
    #         logger.info('mIoU = {:.3f}'.format(mIoU))
    #    #  print(f'mIoU={mIoU:.3f}')