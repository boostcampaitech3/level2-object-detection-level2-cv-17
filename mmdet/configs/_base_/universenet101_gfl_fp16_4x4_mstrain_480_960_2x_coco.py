_base_ = [
    '../universenet/models/universenet101_gfl.py',
    '../datasets/coco_detection_aug.py',
    # '../_base_/datasets/coco_detection_mstrain_480_960.py',
    '../schedules/schedule_2x.py', '../default_runtime.py'
]

data = dict(samples_per_gpu=4)

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# lr_config = dict(warmup_iters=1000)

fp16 = dict(loss_scale=512.)
