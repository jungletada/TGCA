
import os
import copy
import time
import random
import numpy as np
import argparse
import datetime
import hashlib
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
from engine import (
    FixedLengthBatchSampler,
    accumulation_spec,
    linear_scaled_learning_rate,
    train_one_epoch_mctplus,
    train_one_epoch_mctta,
)
from datasets_cam import build_dataset

import models.srmct
import models.mct_adapter
import models.mctformer_plus
from models.mctformer_plus import (
    MCTformerPlus,
    adapt_deit_checkpoint_for_mctformerplus,
    get_mctformerplus_spec,
    model_spec_from_instance,
    resolve_mctformerplus_variant,
)
from models.cti_bgt import add_cti_bgt_arguments, adapt_cti_bgt_finetune
from models.tgca import SUPPORTED_MODES
from models.bcss import BCSS_VARIANTS
from models.persistent_semantic import PSL_VARIANTS, parse_interaction_layers


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument(
        '--accum-iter', default=1, type=int,
        help='number of micro-batches accumulated per optimizer update')
    parser.add_argument(
        '--val-batch-size', default=None, type=int,
        help='validation batch size; default preserves legacy 2x micro batch')
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
    parser.add_argument(
        '--finetune', default='auto',
        help='pretrained checkpoint/URL; auto selects the registered DeiT width')
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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _download_cache_path(url):
    return Path(torch.hub.get_dir()) / 'checkpoints' / Path(url).name


def load_model_weight(args, model):
    """Load model weights from a checkpoint or URL.

    Args:
        args: Command line arguments containing model configuration.
        model: The model instance to load weights into.

    Returns:
        A state dictionary of the model with loaded weights.
    """
    if (isinstance(model, MCTformerPlus)
            and not getattr(model, 'cti_bgt', False)
            and getattr(args, 'bcss_variant', 'e0') == 'e0'
            and getattr(args, 'psl_variant', 'baseline') == 'baseline'):
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
            source_path = _download_cache_path(args.finetune)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            source_path = Path(args.finetune).expanduser().resolve()
        source_state = checkpoint.get('model', checkpoint)
        # Official DeiT has a singleton CLS token and one positional tensor.
        is_deit_source = (
            isinstance(source_state, dict)
            and source_state.get('cls_token') is not None
            and source_state['cls_token'].shape[1] == 1
            and 'pos_embed' in source_state
            and 'pos_embed_cls' not in source_state
        )
        if not is_deit_source:
            raise ValueError(
                'MCTformer+ --finetune expects an official non-distilled DeiT '
                'checkpoint; use --resume for an MCTformer+ training checkpoint'
            )
        adapted, report = adapt_deit_checkpoint_for_mctformerplus(
            checkpoint, model, num_classes=args.nb_classes
        )
        report.update({
            'source_url': model.mctformerplus_pretrained_url,
            'source_argument': args.finetune,
            'cache_path': str(source_path),
            'source_sha256': sha256_file(source_path),
        })
        args.pretrained_load_report = report
        return adapted
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
    if args.batch_size < 1 or args.accum_iter < 1:
        raise ValueError('--batch_size and --accum-iter must be positive')
    if args.val_batch_size is not None and args.val_batch_size < 1:
        raise ValueError('--val-batch-size must be positive when provided')
    mctformerplus_names = {
        'mctformerplus_tiny', 'mctformerplus', 'mctformerplus_base'
    }
    is_mctformerplus = args.model.lower() in mctformerplus_names
    if args.finetune == 'auto':
        if is_mctformerplus:
            args.finetune = get_mctformerplus_spec(args.model)['pretrained_url']
        else:
            # Preserve the historical default for unrelated legacy entry points.
            args.finetune = (
                'https://dl.fbaipublicfiles.com/deit/'
                'deit_small_patch16_224-cd65a155.pth'
            )
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

    batch_contract = accumulation_spec(
        len(dataset_train),
        args.batch_size,
        args.accum_iter,
        utils.get_world_size(),
    )
    args.effective_batch_size = batch_contract['effective_batch_size']
    args.optimizer_updates_per_epoch = batch_contract[
        'optimizer_updates_per_epoch'
    ]
    args.consumed_samples_per_epoch = batch_contract[
        'consumed_samples_per_epoch_global'
    ]
    base_batch_sampler = torch.utils.data.BatchSampler(
        sampler_train, batch_size=args.batch_size, drop_last=True
    )
    fixed_batch_sampler = FixedLengthBatchSampler(
        base_batch_sampler, batch_contract['micro_batches_per_epoch']
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=fixed_batch_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    validation_batch_size = (
        args.val_batch_size
        if args.val_batch_size is not None
        else int(2 * args.batch_size)
    )
    args.val_batch_size = validation_batch_size
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=validation_batch_size,
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

    if is_mctformerplus:
        train_one_epoch = train_one_epoch_mctplus
    else:
        train_one_epoch = train_one_epoch_mctta

    args.pretrained_load_report = None
    if args.finetune:
        checkpoint_model = load_model_weight(args, model)
        strict_pretrain_load = bool(args.pretrained_load_report)
        incompatibility = model.load_state_dict(
            checkpoint_model, strict=strict_pretrain_load
        )
        if strict_pretrain_load and (
                incompatibility.missing_keys or incompatibility.unexpected_keys):
            raise RuntimeError(f'Unexpected strict pretrain load: {incompatibility}')
        if args.psl_variant != 'baseline' and args.finetune.startswith('https'):
            model.initialize_psl_from_backbone()

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    nominal_lr = args.lr
    linear_scaled_lr = linear_scaled_learning_rate(
        nominal_lr, args.effective_batch_size
    )
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    work_space = Path(args.work_space)
    work_space.mkdir(parents=True, exist_ok=True)
    model_spec = model_spec_from_instance(model) if is_mctformerplus else None
    if model_spec is not None:
        (work_space / 'model_spec.json').write_text(
            json.dumps(model_spec, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    if args.pretrained_load_report is not None:
        (work_space / 'pretrained_load_report.json').write_text(
            json.dumps(args.pretrained_load_report, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    pretrained_metadata = {
        'url': (
            args.pretrained_load_report['source_url']
            if args.pretrained_load_report is not None else None
        ),
        'filename': Path(args.finetune).name,
        'sha256': (
            args.pretrained_load_report['source_sha256']
            if args.pretrained_load_report is not None else None
        ),
    }
    training_spec = {
        'seed': args.seed,
        'micro_batch_size': args.batch_size,
        'accum_iter': args.accum_iter,
        'world_size': utils.get_world_size(),
        'effective_batch_size': args.effective_batch_size,
        'nominal_lr': nominal_lr,
        'optimizer_lr': args.lr,
        'epochs': args.epochs,
        'optimizer_updates_per_epoch': args.optimizer_updates_per_epoch,
        'consumed_samples_per_epoch': args.consumed_samples_per_epoch,
        'train_dataset_size': len(dataset_train),
        'val_batch_size': args.val_batch_size,
    }
    optimizer_spec = {
        'optimizer': args.opt,
        'weight_decay': args.weight_decay,
        'epsilon': args.opt_eps,
        'betas': args.opt_betas,
        'schedule': args.sched,
        'warmup_epochs': args.warmup_epochs,
        'minimum_lr': args.min_lr,
        **training_spec,
    }
    (work_space / 'optimizer_spec.json').write_text(
        json.dumps(optimizer_spec, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    def checkpoint_payload(epoch):
        payload = {
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
            'epoch': epoch,
            'pretrained': pretrained_metadata,
            'training_spec': training_spec,
        }
        if model_spec is not None:
            payload['model_spec'] = model_spec
        return payload

    session_name = 'Classification Training'
    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    utils.logger_info(logger_name=session_name, log_path=os.path.join(
                      args.log_dir, f'train-{log_time}-{args.dataset}.log'))
    logger = logging.getLogger(session_name)

    logger.info(vars(args))
    logger.info(f'number of params:{n_parameters}')

    max_accuracy = 0.0
    epoch_runtime = []
    training_loop_started = time.perf_counter()
    maximum_training_allocated = 0
    maximum_training_reserved = 0
    for epoch in range(args.start_epoch, args.epochs):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        train_started = time.perf_counter()
        train_stats = train_one_epoch(
            model, 
            data_loader_train,
            optimizer, 
            device, 
            epoch, 
            loss_scaler,
            args.clip_grad,
            args=args)
        train_seconds = time.perf_counter() - train_started
        if device.type == 'cuda':
            training_peak_allocated = torch.cuda.max_memory_allocated(device)
            training_peak_reserved = torch.cuda.max_memory_reserved(device)
            maximum_training_allocated = max(
                maximum_training_allocated, training_peak_allocated
            )
            maximum_training_reserved = max(
                maximum_training_reserved, training_peak_reserved
            )
        else:
            training_peak_allocated = 0
            training_peak_reserved = 0

        lr_scheduler.step(epoch)

        evaluation_started = time.perf_counter()
        test_stats = evaluate(data_loader_val, model, device)
        evaluation_seconds = time.perf_counter() - evaluation_started
        epoch_runtime.append({
            'epoch': epoch,
            'training_seconds': train_seconds,
            'evaluation_seconds': evaluation_seconds,
            'total_seconds': train_seconds + evaluation_seconds,
            'consumed_training_images': args.consumed_samples_per_epoch,
            'optimizer_updates': args.optimizer_updates_per_epoch,
            'training_images_per_second': (
                args.consumed_samples_per_epoch / train_seconds
            ),
            'optimizer_updates_per_second': (
                args.optimizer_updates_per_epoch / train_seconds
            ),
            'training_peak_allocated_bytes': training_peak_allocated,
            'training_peak_reserved_bytes': training_peak_reserved,
        })
        if test_stats["mAP"] > max_accuracy:
            torch.save(
                checkpoint_payload(epoch),
                os.path.join(args.work_space, f'{args.model}_best.pth'),
            )
        max_accuracy = max(max_accuracy, test_stats["mAP"])

        log_stats = {'epoch': epoch,
                     **{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},}

        if utils.is_main_process():
            logger.info(
                f'mAP on the {len(dataset_val)} eval images: {test_stats["mAP"] * 100:.2f}%\n' +
                f'Max mAP: {max_accuracy * 100:.2f}%\n' + json.dumps(log_stats)
            )

    torch.save(
        checkpoint_payload(args.epochs - 1),
        work_space / f'{args.model}_final.pth',
    )
    total_training_seconds = sum(
        item['training_seconds'] for item in epoch_runtime
    )
    total_optimizer_updates = sum(
        item['optimizer_updates'] for item in epoch_runtime
    )
    total_consumed_images = sum(
        item['consumed_training_images'] for item in epoch_runtime
    )
    runtime = {
        'available': True,
        'measurement_scope': 'current training process',
        'device': str(device),
        'epochs_completed': len(epoch_runtime),
        'wall_seconds_train_and_validation': (
            time.perf_counter() - training_loop_started
        ),
        'training_seconds': total_training_seconds,
        'evaluation_seconds': sum(
            item['evaluation_seconds'] for item in epoch_runtime
        ),
        'mean_epoch_seconds_train_and_validation': float(np.mean([
            item['total_seconds'] for item in epoch_runtime
        ])),
        'training_images_per_second': (
            total_consumed_images / total_training_seconds
        ),
        'optimizer_updates_per_second': (
            total_optimizer_updates / total_training_seconds
        ),
        'training_peak_allocated_bytes': maximum_training_allocated,
        'training_peak_reserved_bytes': maximum_training_reserved,
        'epoch_measurements': epoch_runtime,
    }
    (work_space / 'training_runtime.json').write_text(
        json.dumps(runtime, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.log_dir = os.path.join(args.work_space, args.log_dir)
    Path(args.work_space).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
