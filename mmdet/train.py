# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
from utils import increment_path
import os

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',default = '/opt/ml/detection/_boost_/configs/_base_/faster_rcnn_r50_fpn_1x_coco.py', help='train config file path')
    parser.add_argument('--workdir', default = './work_dirs',help='the root dir to save logs and models about each experiment')
   
    parser.add_argument(
        '--no-validate',
        action='store_true',
        dest = 'no_validate',
        help='whether not to evaluate the checkpoint during training')

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--tags', nargs='+', default=[],
        help ='record your experiment speical keywords into tags list'
        '--tags batch_size=16 swin_cascasde'
        "dont use white space in specific tag") 

    parser.add_argument('--kfold',dest = 'kfold', action='store_true', help='wheter use K-fold Cross-Validation') 
    parser.add_argument('--no-kfold', dest = 'kfold',  action='store_false')
    parser.set_defaults(kfold=True)
    # use : command --kfold -> kfold = True
    # default kfold args.kfold is True 
    # if you don't want kfold, use command option --no-kfold

    parser.add_argument('--fold', type=int, default=1, help='if no kfold cross validation, you must set fold number')

    parser.add_argument('--wandb', dest = 'wandb', action='store_true', help='wandb logging')
    parser.add_argument('--no-wandb', dest = 'wandb',  action='store_false')
    parser.set_defaults(wandb=True)
    
    parser.add_argument('--checkpoint_path',  help='if you want to use pretrained detector, write your path')


    # config options
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args, unknown = parser.parse_known_args()

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args



def main(fold, args ):

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set gpu_id
    cfg.gpu_ids = [0]

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # exp_num path
    workdir = increment_path(os.path.join(args.workdir, 'exp'))

    # create work_dir -> ./workdir if you already set, comment out this line
    mmcv.mkdir_or_exist(osp.abspath(workdir))

    # dump config
    # cfg.dump(osp.join(workdir, osp.basename(args.config)))

    cfg.log_config['hooks'][1]['init_kwargs']['config'] = cfg    

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(workdir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)


    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {True}')
    set_random_seed(seed, deterministic=True)
    cfg.seed = seed

    
    # train_kfold
    if args.no_validate: # 즉, no_validate를 하게 되면 validate안쓰면 무조건 all data로 학습시킨다는 의미.
        cfg.data.train.ann_file = '/opt/ml/detection/dataset/train.json'
        fold = '_all_data'
    
    else:
        # if mix dataset input, dataset ann_file is modified
        if 'ann_file' in cfg.data.train.keys():
            cfg.data.train.ann_file = cfg.json_root + f'train_{fold}.json' 
        else:
            cfg.data.train.dataset.ann_file = cfg.json_root + f'train_{fold}.json' 
        cfg.data.val.ann_file = cfg.json_root + f'val_{fold}.json' 
        
    cfg.work_dir = workdir + f'/Fold{fold}'

    # wandb kfold setting
    cfg.log_config['hooks'][1]['init_kwargs']['group'] = workdir.split('/')[-1]
    cfg.log_config['hooks'][1]['init_kwargs']['job_type'] = f'Fold{fold}'
    cfg.log_config['hooks'][1]['init_kwargs']['name'] = workdir.split('/')[-1] + f'Fold{fold}'
    cfg.log_config['hooks'][1]['init_kwargs']['tags'] = args.tags #args를 그냥 보내서 바뀐 것들은 이걸로 표현해도 나쁘진 않을 듯.
    # 만약 wandb의 config는 맞음 그게


    if not args.wandb : # args.wandb is False -> wandb don't work maybe default = True
        cfg.log_config['hooks']=[dict(type='TextLoggerHook')]
       

    # build dataset & model
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
        )
    model.init_weights()

    if args.checkpoint_path:
        load_checkpoint(model, args.checkpoint_path, map_location='cpu')

    train_detector(
        model,
        datasets,
        cfg,
        validate=(not args.no_validate))


if __name__ == '__main__':
    args = parse_args()

    if args.kfold == True and args.no_validate == False :
        num_folds = 5
        for fold in range(1,num_folds+1):
            main(fold, args)
    else:   
        fold = args.fold
        main(fold, args)



