import io
import os
import sys
import time

import datetime
import logging
import torch
import torch.distributed as dist
import torch.nn.functional as F
from collections import defaultdict, deque

from net.mctformer import MCTformerV2Cam
from net.mctformer_plus import MCTformerPlusCam
from net.mct_adapter import mcta_cam
from net.srmct import SRMCTformerCam


def data_mkdir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
  

def get_dataset_dir(data_set):
    if data_set.lower().__contains__('voc'):
        dataset_dir = 'voc'
    elif data_set.lower().__contains__('coco'):
        dataset_dir = 'coco'
    else:
        raise NotImplementedError    
    return dataset_dir


# def get_dataset_imglist(data_set):
#     if data_set.lower().__contains__('voc'):
#         imglist = 'configs/voc12/val_id.txt'
#     elif data_set.lower().__contains__('coco'):
#         imglist = 'configs/coco/val_id.txt'
#     else:
#         raise NotImplementedError
#     return imglist


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),
           *args, **kwargs)


def logger_info(logger_name, log_path='default_logger.log'):
    ''' 
    Set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('---------Log Handlers exists!---------')
    else:
        print('---------Log Handlers setup!---------')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='w')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)


def load_model_weight(args, model):
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
    except: 
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
        extra_tokens = ckpt_pos_embed[:, :num_extra_tokens].repeat(1, args.nb_classes, 1)
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
        new_cls_token = cls_token_checkpoint.repeat(1, args.nb_classes, 1)
        checkpoint_model['cls_token'] = new_cls_token
    
    return checkpoint_model


def load_model_weight_onecls(args, model):
    model_npatches = model.patch_embed.num_patches
    num_extra_tokens = 1
    if args.finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.finetune, map_location='cpu', check_hash=True)
    else: 
        checkpoint = torch.load(args.finetune, map_location='cpu')
        
    try: checkpoint_model = checkpoint['model']
    except: checkpoint_model = checkpoint
        
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    ckpt_pos_embed = checkpoint_model['pos_embed']
    embedding_size = ckpt_pos_embed.shape[-1]
    original_size = int((ckpt_pos_embed.shape[-2] - num_extra_tokens) ** 0.5)
    
    extra_tokens = ckpt_pos_embed[:, :num_extra_tokens]
    checkpoint_model['pos_embed_cls'] = extra_tokens
    
    pos_tokens = ckpt_pos_embed[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape( # (1, Hp, Wp, C)->(1, C, Hp, Wp)
        -1, original_size, original_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = F.interpolate(
            input=pos_tokens,
            size=(model.Hp, model.Wp),
            mode='bicubic',
            align_corners=False)  
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model['pos_embed_pat'] = pos_tokens

    # if args.finetune.startswith('https'):
    #     cls_token_checkpoint = checkpoint_model['cls_token']
    #     new_cls_token = cls_token_checkpoint.repeat(1, args.nb_classes, 1)
    #     checkpoint_model['cls_token'] = new_cls_token
    
    return checkpoint_model


def create_cam_model(args):
    if 'mctformerv2' in args.model.lower():
        model = MCTformerV2Cam(
            num_classes=args.num_classes,
            input_size=args.input_size)
        
    elif 'mcta' in args.model.lower():
        model = mcta_cam(
            num_classes=args.num_classes,
            input_size=args.input_size)
        
    elif 'mctformerplus' in args.model.lower():
        model = MCTformerPlusCam(
            num_classes=args.num_classes,
            input_size=args.input_size)
        
    elif 'srmctformer' in args.model.lower():
        model = SRMCTformerCam(
            num_classes=args.num_classes,
            input_size=args.input_size)
    else:
        raise NotImplementedError
    
    print(f'Using {args.model} for making class activation maps.')
    
    return model
  
    
class logger_print(object):
    '''
    # ===============================
    # print to file and std_out simultaneously
    # ===============================
    '''
    def __init__(self, log_path="default.log"):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a')

    def write(self, message):
        if is_main_process():
            self.terminal.write(message)
            self.log.write(message)  # write the message

    def flush(self):
        pass


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, rank=0):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}']
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if rank == 0:
            print('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def str2bool(v):
    if v.lower() in ('yes','true','t','y','1','True'):
        return True
    elif v.lower() in ('no','false','f','n','0','False'):
        return False