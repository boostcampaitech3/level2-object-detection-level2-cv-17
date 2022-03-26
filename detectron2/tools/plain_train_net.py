#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer, Checkpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2 import model_zoo
import wandb
wandb.init()
logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        print('EVALUATOR!!!>>>>>>>>>>>>>',evaluator)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    wandb.log({'bbox_AP':results['bbox']['AP'], 'bbox_AP50':results['bbox']['AP50'],'bbox_AP75':results['bbox']['AP75'],'bbox_APs':results['bbox']['APs'],'bbox_APm':results['bbox']['APm'],
    'bbox_APl':results['bbox']['APl'],'AP-General trash':results['bbox']['AP-General trash'],'AP-Paper':results['bbox']['AP-Paper'],'AP-Paper pack':results['bbox']['AP-Paper pack'],'AP-Metal':results['bbox']['AP-Metal'],
    'AP-Glass':results['bbox']['AP-Glass'],'AP-Plastic':results['bbox']['AP-Plastic'],'AP-Styrofoam':results['bbox']['AP-Styrofoam'],'AP-Plastic bag':results['bbox']['AP-Plastic bag'],'AP-Battery':results['bbox']['AP-Battery'],'AP-Clothing':results['bbox']['AP-Clothing']})
    return results


def do_train(cfg, model, best_AP50, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            # print('>>loss_dict_reduced : ', loss_dict_reduced)    >>loss_dict_reduced :  {'loss_cls': 2.3326451778411865, 'loss_box_reg': 0.4775199294090271, 'loss_rpn_cls': 0.12368661165237427, 'loss_rpn_loc': 0.030212724581360817}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # print('>>losses_reduced : ', losses_reduced)   >>losses_reduced :  2.8170977495610714
            wandb.log({'train_cls_loss':loss_dict_reduced['loss_cls'], 'train_box_reg_loss':loss_dict_reduced['loss_box_reg'], 
                        'train_rpn_cls_loss':loss_dict_reduced['loss_rpn_cls'], 'train_rpn_loc_loss':loss_dict_reduced['loss_rpn_loc']})
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                rst = do_test(cfg, model)
                print('>>>RESULT')
                if rst['bbox']['AP50'] > best_AP50:
                    print('>>>> Best Test AP50 Score!!!')
                    best_AP50 = rst['bbox']['AP50']
                    checkpointer.save('AP50_best')

                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'))
    # cfg.merge_from_list(args.opts)
    fold = 1
      # if you don't like any of the default setup, write your own setup code

    try:
        register_coco_instances('coco_trash_train', {}, '/opt/ml/detection/baseline/detectron2/stratified_kfold/cv_train_'+str(fold)+'.json', '/opt/ml/detection/dataset/')
    except AssertionError:
        pass

    try:
        register_coco_instances('coco_trash_test', {}, '/opt/ml/detection/baseline/detectron2/stratified_kfold/cv_val_'+str(fold)+'.json', '/opt/ml/detection/dataset/')
    except AssertionError:
        pass
    MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                            "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    cfg.DATASETS.TEST = ('coco_trash_test',)
    cfg.DATALOADER.NUM_WOREKRS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 20000 # 20 epoch
    cfg.SOLVER.STEPS = (8000,12000)
    cfg.SOLVER.GAMMA = 0.005
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000
    cfg.OUTPUT_DIR = './detectron2/output_eval'

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

    cfg.TEST.EVAL_PERIOD = 1000
    default_setup(
        cfg, args
    )
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup(args)
    # print(cfg)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    best_AP50 = 0
    do_train(cfg, model, best_AP50, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
