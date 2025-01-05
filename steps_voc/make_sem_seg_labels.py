import os
import torch
import imageio
import numpy as np
from tqdm import tqdm
import importlib
from misc import imutils
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import voc12.dataloader
from misc import torchutils, indexing
cudnn.enabled = True


def _work(process_id, model, dataset, args):
    n_gpus = torch.cuda.device_count()
    databin = dataset[process_id]
    data_loader = DataLoader(
        databin, shuffle=False, 
        num_workers=args.num_workers // n_gpus, 
        pin_memory=False)

    with torch.no_grad(), cuda.device(process_id):
        model.cuda()
        
        for iter, pack in enumerate(tqdm(data_loader, position=process_id, desc=f'[PID{process_id}]')):
            img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
            original_size = pack['size']
            edge, dp = model(pack['img'][0].cuda(non_blocking=True))
            
            cam_dict = np.load(f"{args.cam_out_dir}/{img_name}.npy", allow_pickle=True).item()
            cams = cam_dict['cam']
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cam_downsized_values = cams.cuda()

            rw = indexing.propagate_to_edge(
                cam_downsized_values, 
                edge, 
                beta=args.beta, 
                exp_times=args.exp_times, 
                radius=5)
            
            rw_up = F.interpolate(
                input=rw, 
                size=original_size, 
                mode='bilinear', 
                align_corners=False).squeeze(1)
            
            rw_up = rw_up / torch.max(rw_up)
            
            # rw_up_npy = rw_up.cpu().numpy()
            # rw_dict = {"keys": cam_dict['keys'], "high_res": rw_up_npy}
            # np.save(os.path.join(args.sem_seg_npy_dir, img_name + '.npy'), rw_dict)
            
            # bg_score = np.power(1 - np.max(rw_up_npy, axis=0, keepdims=True), 1.2)
            # rw_up_npy = np.concatenate((bg_score, rw_up_npy), axis=0)
            # rw_pred = np.argmax(rw_up_npy, axis=0)
            
            rw_up_bg = F.pad(rw_up, (0, 0, 0, 0, 1, 0), value=args.sem_seg_bg_thres)
            rw_pred = torch.argmax(rw_up_bg, dim=0).cpu().numpy()

            rw_pred = keys[rw_pred]
            imageio.imsave(os.path.join(args.sem_seg_out_dir, img_name + '.png'), rw_pred.astype(np.uint8))

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5*iter+1)//(len(databin) // 20)), end='')


def run(args):
    model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')()
    model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
    model.eval()

    n_gpus = torch.cuda.device_count()

    from voc12.dataloader import VOC12ClassificationDatasetMSF
    dataset = VOC12ClassificationDatasetMSF(
        args.infer_list,
        voc12_root=args.voc12_root,
        scales=(1.0,),
        make_seg=True)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[", end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print("]")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    pass
