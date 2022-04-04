_base_ =[
    '../schedules/schedule_1x.py', '../default_runtime.py'
]

img_scale = (640, 640)

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=10, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
dataset_type = 'CocoDataset'
data_root='/content/dataset/'
json_root = '/content/stratified_kfold/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")


# train_pipeline = [
#     # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
#     dict(
#         type='RandomAffine',
#         scaling_ratio_range=(0.1, 2),
#         border=(-img_scale[0] // 2, -img_scale[1] // 2)),
#     # dict(
#     #     type='MixUp',
#     #     img_scale=img_scale,
#     #     ratio_range=(0.8, 1.6),
#     #     pad_val=114.0),
#     dict(type='CutOut', 
#     n_holes=7,
#     cutout_shape=[(4, 4), (4, 8), (8, 4),
#                        (8, 8), (16, 8), (8, 16),
#                        (16, 16), (16, 32), (32, 16) ]),
#     dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
#                  min_crop_size=0.3,
#                  bbox_clip_border=True),
#     dict(type='YOLOXHSVRandomAug'),
#     dict(type='RandomFlip', flip_ratio=0.8),
#     # According to the official implementation, multi-scale
#     # training is not considered here but in the
#     # 'mmdet/models/detectors/yolox.py'.
#     dict(type='Resize', img_scale=img_scale, keep_ratio=True),
#     dict(
#         type='Pad',
#         pad_to_square=True,
#         # If the image is three-channel, the pad value needs
#         # to be set separately for each channel.
#         pad_val=dict(img=(114.0, 114.0, 114.0))),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=json_root +  'train_1.json',
        img_prefix=data_root,
        classes = classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        # filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=json_root + 'val_1.json',
        img_prefix=data_root ,
        classes = classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root ,
        classes = classes,
        pipeline=test_pipeline))


max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 10

# learning policy
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
checkpoint_config = dict(interval=interval)
evaluation = dict(interval=interval, metric='bbox',    
                classwise=True,
                iou_thrs=[0.50],
                metric_items=['mAP','mAP_s','mAP_m','mAP_l'])

log_config = dict(interval=200)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

