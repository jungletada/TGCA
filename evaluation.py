import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import logging
from utils import logger_info


categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']


def do_python_eval(
        pred_folder, 
        gt_folder, 
        name_list, 
        num_classes=21, 
        input_type='png', 
        threshold=1.0, 
        logger_=None):
    TP, P, T = [], [], []
    for i in range(num_classes):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    def compare(start, step, TP, P, T, input_type, threshold):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]
            if input_type == 'png':
                predict_file = os.path.join(pred_folder,'%s.png'%name)
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
                if num_classes == 81:
                    predict = predict - 91
                    
            elif input_type == 'npy':
                predict_file = os.path.join(pred_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((num_classes, h, w), np.float32)
                for key in predict_dict.keys():
                    tensor[key + 1] = predict_dict[key]
                tensor[0, :, :] = threshold 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            gt_file = os.path.join(gt_folder,'%s.png' % name)
            gt = np.array(Image.open(gt_file))
            cal = gt < 255
            
            mask = (predict == gt) * cal
      
            for i in range(num_classes):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,input_type,threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    
    for i in range(num_classes):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        
    loglist = {}
    for i in range(num_classes):
        loglist[categories[i]] = IoU[i] * 100
               
    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    fp = np.mean(np.array(FP_ALL))
    loglist['FP'] = fp * 100
    fn = np.mean(np.array(FN_ALL))
    loglist['FN'] = fn * 100
    
    if logger_ is not None:
        for i in range(num_classes):
            logger_.info('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
            
        logger_.info('======================================================')
        logger_.info(f'FP = {fp * 100}, FN = {fn * 100}')
        logger_.info('%11s:%7.3f%%'%('mIoU', miou * 100))
        
    return loglist


def dict_to_message(dictionary):
    msg = ''
    for key, value in dictionary.items():
        sub = f'{key}:{value}\n'
        msg += sub
    return msg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_space", default="results/MCTG", type=str)
    parser.add_argument("--infer_list", default='configs/voc12/train_id.txt', type=str)
    parser.add_argument("--predict_dir", default='results/MCTG/pseudo_mask', type=str)
    parser.add_argument("--gt_dir", default='datasets/VOCdevkit/VOC2012/SegmentationClassAug', type=str)
    parser.add_argument('--log_file', default='eval_pseudo_mask.log',type=str)
    
    parser.add_argument('--type', default='png', choices=['npy', 'png'], type=str)
    parser.add_argument('--threshold', default=None, type=float)
    parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument('--num_classes', default=21, type=int)
    
    parser.add_argument('--start', default=30, type=int)
    parser.add_argument('--end', default=60, type=int)
    
    args = parser.parse_args()

    if args.type == 'npy':
        assert args.threshold is not None or args.curve
    df = pd.read_csv(args.infer_list, names=['filename'])
    name_list = df['filename'].values

    session_name = 'evaluation'
    args.log_file = os.path.join(args.work_space, args.log_file)
    logger_info(logger_name=session_name, log_path=args.log_file)
    logger = logging.getLogger(session_name)
    logger.info(f"Logs save path: {args.log_file}")
    
    if not args.curve:
        loglist = do_python_eval(
            args.predict_dir, 
            args.gt_dir, 
            name_list, 
            args.num_classes, 
            args.type, 
            args.threshold, 
            logger_=logger)
       
        # logger.info(dict_to_message(loglist))

    else:
        mIoU_curves = []
        max_mIoU, best_thr = 0.0, 0.0
        for i in range(args.start, args.end):
            threshold = i / 100.0
            loglist = do_python_eval(
                args.predict_dir, 
                args.gt_dir, 
                name_list, 
                args.num_classes, 
                args.type, 
                threshold)
            mIoU_curves.append(loglist['mIoU'])
            logger.info('[%d/%d] background score: %.3f\tmIoU: %.3f%%'%(
                i, args.end, threshold, loglist['mIoU']))
            
            if loglist['mIoU'] > max_mIoU:
                max_mIoU = loglist['mIoU']
                best_thr = threshold
            else:
                break
        logger.info('Best background score: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))