
import os
import copy
import time
import random
import numpy as np
import argparse
import datetime
import json
import logging
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

import utils
from engine import evaluate
from engine import train_one_epoch_mctplus, train_one_epoch_mctta
from datasets_cam import build_dataset

import models.srmct
import models.mct_adapter
import models.mctformer_plus
from models.cti_bgt import add_cti_bgt_arguments, adapt_cti_bgt_finetune
from models.tgca import SUPPORTED_MODES
from models.bcss import BCSS_VARIANTS
from models.persistent_semantic import PSL_VARIANTS, parse_interaction_layers


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=45, type=int)

    # Model parameters
    parser.add_argument('--model', default='mcta', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=448, type=int, help='images input size')
    parser.add_argument(
        '--attention-normalization',
        default='vanilla',
        choices=sorted(SUPPORTED_MODES),
        help='attention normalization used in the shared MCTformer+ attention path')
    parser.add_argument(
        '--attention-gamma', default=1.0, type=float,
        help='key-group count correction exponent (TGCA modes only)')
    parser.add_argument('--bcss-variant', default='e0', choices=tuple(BCSS_VARIANTS),
                        help='BCSS screening or prespecified debug variant')
    parser.add_argument('--bcss-num-background-slots', default=1, type=int)
    parser.add_argument('--bcss-tau', default=0.5, type=float)
    parser.add_argument('--bcss-beta', default=0.5, type=float)
    parser.add_argument('--bcss-cls-threshold', default=0.5, type=float)
    parser.add_argument('--bcss-lambda-fg', default=0.5, type=float)
    parser.add_argument('--bcss-lambda-bg', default=0.1, type=float)
    parser.add_argument('--bcss-semantic-temperature', default=1.0, type=float)
    parser.add_argument('--psl-variant', default='baseline', choices=PSL_VARIANTS)
    parser.add_argument(
        '--psl-interaction-layers', default=(11,),
        type=parse_interaction_layers,
        help='zero-based comma-separated semantic read/write layers')
    parser.add_argument('--psl-relation-dim', default=384, type=int)
    parser.add_argument('--psl-num-background-latents', default=1, type=int)
    add_cti_bgt_arguments(parser)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth', 
                        help='finetune from checkpoint')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--work_space", default="results/mcta", type=str)
    parser.add_argument('--log_dir', default='log_dir', type=str, help='log dir to save the results')

    # Dataset parameters
    parser.add_argument('--dataset', default='', type=str, help='name of dataset')
    parser.add_argument('--voc12_root', default='data/VOCdevkit/VOC2012', type=str, help='VOC12 dataset path')
    parser.add_argument("--coco_root", default='data/MSCOCO', type=str, help="Path to MSCOCO")
    parser.add_argument("--train_list", default="train_aug_id.txt", type=str, 
                        help='train_id.txt or train_aug_id.txt')
    parser.add_argument("--val_list", default="val_id.txt", type=str,
                        help='validation image-id list')
    parser.add_argument('--checkpoint', default='', help='checkpoint for generating maps')

    parser.add_argument('--seed', default=None, type=int)

    return parser


def load_model_weight(args, model):
    """Load model weights from a checkpoint or URL.

    Args:
        args: Command line arguments containing model configuration.
        model: The model instance to load weights into.

    Returns:
        A state dictionary of the model with loaded weights.
    """
    if getattr(model, 'cti_bgt', False):
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
        return adapt_cti_bgt_finetune(checkpoint, model)
    nc = args.nb_classes
    model_npatches = model.patch_embed.num_patches
    if args.finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.finetune, map_location='cpu', check_hash=True)
        num_extra_tokens = 1
    else: 
        checkpoint = torch.load(args.finetune, map_location='cpu')
        num_extra_tokens = model.pos_embed.shape[-2] - model_npatches
    try: 
        checkpoint_model = checkpoint['model']
    except KeyError:  # Specify the exception type
        checkpoint_model = checkpoint
        
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    ckpt_pos_embed = checkpoint_model['pos_embed']
    embedding_size = ckpt_pos_embed.shape[-1]

    original_size = int((ckpt_pos_embed.shape[-2] - num_extra_tokens) ** 0.5)
    if args.finetune.startswith('https'):
        extra_tokens = ckpt_pos_embed[:, :num_extra_tokens].repeat(1, nc, 1)
    else:
        extra_tokens = ckpt_pos_embed[:, :num_extra_tokens]
    pos_tokens = ckpt_pos_embed[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape( # (1, Hp, Wp, C)->(1, C, Hp, Wp)
        -1, original_size, original_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(
            input=pos_tokens,
            size=(model.Hp, model.Wp),
            mode='bicubic',
            align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

    checkpoint_model['pos_embed_cls'] = extra_tokens
    checkpoint_model['pos_embed_pat'] = pos_tokens

    if args.finetune.startswith('https'):
        cls_token_checkpoint = checkpoint_model['cls_token']
        new_cls_token = cls_token_checkpoint.repeat(1, nc, 1)
        checkpoint_model['cls_token'] = new_cls_token

    return checkpoint_model


def main(args):
    device = torch.device(args.device)
    if args.seed is None:
        args.seed = random.randint(0, 29510)
    seed = args.seed
    # cudnn.benchmark = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    dataset_train, args.nb_classes = build_dataset(
        is_train=True, make_cam=False, args=args)

    val_args = copy.copy(args)
    val_args.train_list = args.val_list
    dataset_val, _ = build_dataset(
        is_train=False, make_cam=False, args=val_args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(2 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        input_size=args.input_size,
        attention_normalization=args.attention_normalization,
        attention_gamma=args.attention_gamma,
        bcss_variant=args.bcss_variant,
        bcss_num_background_slots=args.bcss_num_background_slots,
        bcss_tau=args.bcss_tau,
        bcss_beta=args.bcss_beta,
        bcss_cls_threshold=args.bcss_cls_threshold,
        bcss_lambda_fg=args.bcss_lambda_fg,
        bcss_lambda_bg=args.bcss_lambda_bg,
        bcss_semantic_temperature=args.bcss_semantic_temperature,
        psl_variant=args.psl_variant,
        psl_interaction_layers=args.psl_interaction_layers,
        psl_relation_dim=args.psl_relation_dim,
        psl_num_background_latents=args.psl_num_background_latents,
        cti_bgt=args.cti_bgt,
        cti_bgt_weight=args.cti_bgt_weight,
        cti_bgt_n_layers=args.cti_bgt_n_layers,
        cti_bgt_affinity_start=args.cti_bgt_affinity_start)

    # Variant-specific parameter initialization consumes different amounts of
    # RNG. Reset before optimization so augmentation, dropout, and sampling use
    # an identical stochastic stream in every matched screening run.
    training_seed = seed + 1000003
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    np.random.seed(training_seed % (2 ** 32))
    random.seed(training_seed)

    if "mctformerplus" in args.model:
        train_one_epoch = train_one_epoch_mctplus
    else:
        train_one_epoch = train_one_epoch_mctta

    if args.finetune:
        checkpoint_model = load_model_weight(args, model)
        model.load_state_dict(checkpoint_model, strict=False)
        if args.psl_variant != 'baseline' and args.finetune.startswith('https'):
            model.initialize_psl_from_backbone()

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    work_space = Path(args.work_space)
    session_name = 'Classification Training'
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    utils.logger_info(logger_name=session_name, log_path=os.path.join(
                      args.log_dir, f'train-{time}-{args.dataset}.log'))
    logger = logging.getLogger(session_name)

    logger.info(vars(args))
    logger.info(f'number of params:{n_parameters}')

    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, 
            data_loader_train,
            optimizer, 
            device, 
            epoch, 
            loss_scaler,
            args.clip_grad,
            args=args)

        lr_scheduler.step(epoch)

        test_stats = evaluate(data_loader_val, model, device)
        if test_stats["mAP"] > max_accuracy:
            torch.save({
                        'model': model.state_dict(),
                        'bcss': {
                            'variant': args.bcss_variant,
                            'num_background_slots': args.bcss_num_background_slots,
                            'tau': args.bcss_tau,
                            'beta': args.bcss_beta,
                            'class_threshold': args.bcss_cls_threshold,
                            'lambda_fg': args.bcss_lambda_fg,
                            'lambda_bg': args.bcss_lambda_bg,
                            'semantic_temperature': args.bcss_semantic_temperature,
                            'foreground_anchor_mode': (
                                'ownership_mass_scaled'
                                if args.bcss_variant == 'e4_mass'
                                else 'spatial_normalized'
                            ),
                        },
                        'attention_normalization': {
                            'mode': args.attention_normalization,
                            'gamma': args.attention_gamma,
                            'relation_bias': args.attention_normalization == 'tgca_bias',
                        },
                        'cti_bgt': model.cti_bgt_configuration(),
                        'psl': model.psl_configuration(),
                        'epoch': epoch,
                       },
                       os.path.join(args.work_space, f'{args.model}_best.pth'))
        max_accuracy = max(max_accuracy, test_stats["mAP"])

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},}

        if utils.is_main_process():
            logger.info(
                f'mAP on the {len(dataset_val)} eval images: {test_stats["mAP"] * 100:.2f}%\n' +
                f'Max mAP: {max_accuracy * 100:.2f}%\n' + json.dumps(log_stats)
            )

    torch.save({
        'model': model.state_dict(),
        'bcss': {
            'variant': args.bcss_variant,
            'num_background_slots': args.bcss_num_background_slots,
            'tau': args.bcss_tau,
            'beta': args.bcss_beta,
            'class_threshold': args.bcss_cls_threshold,
            'lambda_fg': args.bcss_lambda_fg,
            'lambda_bg': args.bcss_lambda_bg,
            'semantic_temperature': args.bcss_semantic_temperature,
            'foreground_anchor_mode': (
                'ownership_mass_scaled'
                if args.bcss_variant == 'e4_mass'
                else 'spatial_normalized'
            ),
        },
        'attention_normalization': {
            'mode': args.attention_normalization,
            'gamma': args.attention_gamma,
            'relation_bias': args.attention_normalization == 'tgca_bias',
        },
        'psl': model.psl_configuration(),
        'cti_bgt': model.cti_bgt_configuration(),
        'epoch': args.epochs - 1,
    }, work_space / f'{args.model}_final.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.log_dir = os.path.join(args.work_space, args.log_dir)
    Path(args.work_space).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
