import cv2
import os
import math
import sys
import six
import torch
import utils
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Iterable
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import average_precision_score
# import seaborn as sns


def train_one_epoch_proposed(
    args,model, data_loader, optimizer, device, epoch,
    loss_scaler, max_norm, set_training_mode=True, rank=0):

    print_freq = 10
    model.train(set_training_mode)
    criterion = nn.MultiLabelSoftMarginLoss()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, rank=rank):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)

            cls_loss = criterion(outputs[0], targets)
            metric_logger.update(cls_loss=cls_loss.item())
            
            patch_loss = criterion(outputs[1], targets)
            metric_logger.update(pat_loss=patch_loss.item())
            
            total_loss = args.cls_weight * cls_loss + patch_loss
            
        loss_value = total_loss.item()

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(total_loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    if dist.get_rank() == 0:
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  


def train_one_epoch_next(
    args,model, data_loader, optimizer, device, epoch,
    loss_scaler, max_norm, set_training_mode=True, rank=0):

    print_freq = 10
    model.train(set_training_mode)
    criterion = nn.MultiLabelSoftMarginLoss()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, rank=rank):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)

            cls_loss = criterion(outputs[0], targets)
            metric_logger.update(cls_loss=cls_loss.item())
            
            patch_loss = criterion(outputs[1], targets)
            metric_logger.update(pat_loss=patch_loss.item())
            
            total_loss = args.cls_weight * cls_loss + patch_loss
            
        loss_value = total_loss.item()

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(total_loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    if dist.get_rank() == 0:
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_basic(
    args, model, data_loader, optimizer, device, epoch,
    loss_scaler, max_norm, set_training_mode=True, rank=0):

    print_freq = 10
    model.train(set_training_mode)
    criterion = nn.MultiLabelSoftMarginLoss()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, rank=rank):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples)

            total_loss = criterion(outputs[0], targets)
            metric_logger.update(total_loss=total_loss.item())
            
        loss_value = total_loss.item()

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(total_loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    if dist.get_rank() == 0:
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  


def train_one_epoch_mctformerplus(
        args,model,data_loader,optimizer, device,epoch, loss_scaler, 
        max_norm, set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        patch_outputs = None
        c_outputs = None
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs, c_outputs, patch_outputs = outputs

            loss = F.multilabel_soft_margin_loss(outputs, targets)
            metric_logger.update(mct_loss=loss.item())
            
            output_cls_embeddings = F.normalize(c_outputs, dim=-1)  # 12xBxCxD
            scores = output_cls_embeddings @ output_cls_embeddings.permute(0, 1, 3, 2)  # 12xBxCxC

            ground_truth = torch.arange(targets.size(-1), dtype=torch.long, device=device)  # C
            ground_truth = ground_truth.unsqueeze(0).unsqueeze(0).expand(
                c_outputs.shape[0], c_outputs.shape[1], c_outputs.shape[2])  # 12 x B x C
            regularizer_loss = torch.nn.CrossEntropyLoss(reduction='none')(
                scores.permute(1, 2, 3, 0), ground_truth.permute(1, 2, 0))  # B x C x 12
            regularizer_loss = torch.mean(
                torch.mean(torch.sum(regularizer_loss * targets.unsqueeze(-1), dim=-2), dim=-1) / (
                            torch.sum(targets, dim=-1) + 1e-8))
            metric_logger.update(attn_loss=regularizer_loss.item())
            loss = loss + args.cls_weight * regularizer_loss
            
            ploss = F.multilabel_soft_margin_loss(patch_outputs, targets)
            metric_logger.update(pat_loss=ploss.item())
            loss = loss + ploss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    mAP = []
    model.eval() # switch to evaluation mode

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            output = model(images)
            if isinstance(output, (list, tuple)):
                pred = output[0]
            else:
                pred = output
            loss = criterion(pred, target)
            pred = torch.sigmoid(pred)

            mAP_list = compute_mAP(target, pred)
            mAP = mAP + mAP_list
            metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f'* mAP {metric_logger.mAP.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
    return AP


def de_normalize_image(image):
    """
        De-normalize the image to [0, 255]
        image: numpy.array with shape(b, c, h, w)
        return: numpy.array, denormalize image with value between (0, 255)
    """
    # Put the channel dimension to last -> b, h, w, 3
    img_temp = image.permute(0, 2, 3, 1).detach().cpu().numpy()
    original_images = np.zeros_like(img_temp)
    original_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
    original_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
    original_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.
    return original_images


def get_feature_map_size(images, patch_size):
    w_round = images.shape[2] - images.shape[2] % patch_size
    h_round = images.shape[3] - images.shape[3] % patch_size
    w_featmap, h_featmap = w_round // patch_size, h_round // patch_size
    return w_featmap, h_featmap


@torch.no_grad()
def make_cam_ms(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.to(device)
    model.eval() # switch to evaluation mode
    
    index = 0
    img_list_txt = args.infer_list
    img_name_list = open(os.path.join(args.img_list, img_list_txt)).readlines()
    
    # multi-scale dataset: [s1, s1', s2, s2', ...] with batch size = 1, 
    for ms_images, target in metric_logger.log_every(data_loader, print_freq=50):
        # len(ms_images) = 2 since scale = 1, original and flip
        s0_image = ms_images[0].to(device, non_blocking=True) # (b, c, h, w) with b = 1, c = 3
        target = target.to(device, non_blocking=True)         # (b, cls) with b = 1, cls = 20 in VOC
        img_name = img_name_list[index].strip()   
        index += 1

        original_images = de_normalize_image(s0_image) # b x h x w x 3
        H0, W0 = original_images.shape[1:3]
        num_scales = len(ms_images)
        
        with torch.cuda.amp.autocast():
            cam_map_list = [] # class attention maps
            for scale in range(num_scales):
                images = ms_images[scale].to(device, non_blocking=True)
                # obtain the feature map size with  w_featmap and h_featmap
                Hf, Wf = get_feature_map_size(images, args.patch_size)
                 # cls_logits, class-to-patch attention, patch-to-patch attention
                # (B x Cls), (B x Cls x Hp x Wp), (L x B x Np x Np), where Np = Hp x Wp
                
                # 'MCTformerV2' 
                cls_logits, cam_refine = model(
                    images, fuse_layers=args.layer_index)
                
                # elif 'MCTformerPlus' in args.model:
                #     cls_logits, cam_refine, pat2pat = model(
                #         images, 
                #         return_att=True, 
                #         n_layers=args.layer_index)
                #     pat2pat = torch.sum(pat2pat, dim=0) # B x Np x Np
                
                # cls_logits, cam_refine = model(
                #     images, 
                #     return_att=True,
                #     n_layers=args.layer_index)

                # if args.visualize_dir is not None and scale % 2 == 0:
                #     visualize_attention(attn=attn_maps[0], save_name=img_name, args=args)
                
                # resize class-to-patch attentions to original size
                cam_refine = F.interpolate(
                    input=cam_refine, 
                    size=(H0, W0), 
                    mode='bilinear', 
                    align_corners=False)[0]
                
                # (Cls x W0 x H0) (Cls x 1 x 1)
                cam_refine = cam_refine.cpu().numpy() * \
                    target.clone().view(args.nb_classes, 1, 1).cpu().numpy() 
                # flip back the cls_attentions
                if scale % 2 == 1:
                    cam_refine = np.flip(cam_refine, axis=-1)

                cam_map_list.append(cam_refine)

            sum_cam = np.sum(cam_map_list, axis=0)
            sum_cam = torch.from_numpy(sum_cam)
            sum_cam = sum_cam.unsqueeze(0).to(device)   # 1 x Cls x W0 x H0
            # cls_logits = torch.sigmoid(cls_logits)      # 1 x Cls

        for batch in range(images.shape[0]): # for each image
            if (target[batch].sum()) > 0:
                cam_dict = {}
                for cls_index in range(args.nb_classes):
                    if target[batch, cls_index] > 0:
                        cls_score = format(cls_logits[batch, cls_index].cpu().numpy(), '.3f')
                        cam_cls = sum_cam[batch, cls_index, :]
                        cam_cls = (cam_cls - cam_cls.min()) / (cam_cls.max() - cam_cls.min() + 1e-8)
                        cam_cls = cam_cls.cpu().numpy()
                        cam_dict[cls_index] = cam_cls

                        # if args.attention_dir is not None:
                        #     fname = os.path.join(args.attention_dir, f"{img_name}_{cls_index}_{cls_score}.png")
                        #     show_cam_on_image(original_images[batch], cam_cls, fname)

                
                np.save(os.path.join(args.cam_npy_dir, f"{img_name}.npy"), cam_dict)
    
                # if args.out_crf is not None: # refine with crf
                #     image_np = original_images[batch].astype(np.uint8).copy(order='C')
                #     label_la = _crf_with_alpha(cam_dict, args.low_alpha, image_np)
                #     label_ha = _crf_with_alpha(cam_dict, args.high_alpha, image_np)
                #     crf_refine = np.array(list(label_la.values()) + list(label_ha.values()))
                #     crf_refine = np.transpose(crf_refine, (1, 2, 0))
                #     np.save(os.path.join(args.out_crf, f"{img_name}.npy"), crf_refine)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()


def _crf_with_alpha(cam_dict, alpha, orig_img):
    from psa_network.tool.imutils import crf_inference
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])
    n_crf_al = dict()
    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)


def visualize_attention(attn, save_name, args):
    attn = (attn - attn.min()) / (attn.max() - attn.min())
    attn = attn.cpu().numpy()
    Cls = args.nb_classes
    Np = attn.shape[-1] - Cls
    cmap = "RdBu_r"
    fig, axs = plt.subplots(2, 2)
    # 创建一个具有特定网格布局的图形
    fig = plt.figure(figsize=(5, 5))
    cls, np = 20, 80; nl = cls + np
    gs = GridSpec(nl, nl, figure=fig)
    # 添加子图
    ax1 = fig.add_subplot(gs[:cls, :cls])
    ax2 = fig.add_subplot(gs[:cls, cls:])
    ax3 = fig.add_subplot(gs[cls:, :cls])
    ax4 = fig.add_subplot(gs[cls:, cls:])
    # 在每个子图上绘制热力图
    sns.heatmap(attn[:Cls, :Cls], ax=ax1, cbar=False, cmap=cmap)
    sns.heatmap(attn[:Cls, Cls:], ax=ax2, cbar=False, cmap=cmap)
    sns.heatmap(attn[Cls:, :Cls], ax=ax3, cbar=False, cmap=cmap)
    sns.heatmap(attn[Cls:, Cls:], ax=ax4, cbar=False, cmap=cmap)
    
    # 设置子图的标题（可选）
    # axs[0, 0].set_title('Cls2Cls')
    # axs[0, 1].set_title('Cls2Pat')
    # axs[1, 0].set_title('Pat2Cls')
    # axs[1, 1].set_title('Pat2Pat')
    # # 设置第 1 和第 3 个子图的 ylabel
    # ax1.set_ylabel('Class')
    # ax3.set_ylabel('Patch')
    # # 设置第 2 和第 4 个子图的 xlabel
    # ax3.set_xlabel('Class')
    # ax4.set_xlabel('Patch')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)  # 移除刻度线
    # 调整子图间距
    # plt.tight_layout()
    save_path = os.path.join(args.visualize_dir, f'{save_name}_attn_mct.png')
    plt.savefig(save_path, dpi=300)  
    plt.close()
    

def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    """Collect a confusion matrix.

    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.

    Args:
        pred_labels (iterable of numpy.ndarray): See the table in
            :func:`chainercv.evaluations.eval_semantic_segmentation`.
        gt_labels (iterable of numpy.ndarray): See the table in
            :func:`chainercv.evaluations.eval_semantic_segmentation`.

    Returns:
        numpy.ndarray:
        A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
        The :math:`(i, j)` th element corresponds to the number of pixels
        that are labeled as class :math:`i` by the ground truth and
        class :math:`j` by the prediction.

    """
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 0
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError(f'ndim={pred_label.ndim}, ndim={gt_label.ndim}, ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) +
            pred_label[mask], minlength=n_class**2).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return confusion
