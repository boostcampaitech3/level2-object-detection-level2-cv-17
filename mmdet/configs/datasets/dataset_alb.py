# dataset settings
dataset_type = 'CocoDataset'
data_root='/opt/ml/detection/dataset/'
json_root = '/opt/ml/detection/dataset/stratified_kfold/'
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)



# 추가하고 싶은 목록

# - CLAHE
# - channel dropout, channel shuffle
# - Random shadow
# - solarize
# - superpixel
# (CutOut, MinIoURandomCrop, MixUp, Mosaic,
#  PhotoMetricDistortion, RandomAffine, RandomShift )
albu_train_transforms = [
    dict(
    type='OneOf',
    transforms=[
        dict(type='Flip',p=1.0),
        dict(type='HorizontalFlip',p=1.0),
        # dict(type='VerticalFlip ',p=1.0),
        dict(type='RandomRotate90',p=1.0)
    ],
    p=0.5),
    dict(
        type = 'OneOf', # - channel dropout, channel shuffle
        transforms = [
            dict(type='ChannelDropout',p=0.5),
            dict(type='ChannelShuffle', p=0.5)
        ],
    p=0.1),
    dict(type='CLAHE',p=0.1),
    dict(type='RandomShadow',p=0.1),# - Random shadow
    # dict(type='Flip',p=1.0),# - solarize
    # dict(type='Flip',p=1.0),# - superpixel

    # dict(type='RandomResizedCrop',height=512, width=512, scale=(0.5, 1.0), p=0.5), -> 이것도 꼭 써보도록 해라.


    # dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.0, rotate_limit=0, interpolation=1, p=0.5),
    dict(type='RandomBrightnessContrast',brightness_limit=0.1, contrast_limit=0.15, p=0.3),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.1),
    dict(type='GaussNoise', p=0.1),
    dict(
    type='OneOf',
    transforms=[
        dict(type='Blur', p=1.0),
        dict(type='GaussianBlur', p=1.0),
        dict(type='MedianBlur', blur_limit=5, p=1.0),
        dict(type='MotionBlur', p=1.0)
    ],
    p=0.1)

    # dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.0, rotate_limit=0, interpolation=1, p=0.5),
    #     dict(type='RandomBrightnessContrast', brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=0.2),
    #     dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    #     dict(type='ChannelShuffle', p=0.1),
]

# ㅁaugmentation에 안좋은 영향을 주는 게 있을 수 있음.

# (CutOut, MinIoURandomCrop, MixUp, Mosaic,
#   RandomAffine, RandomShift )

# set train_pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True), # Bring Annotation boxes
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.7),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True
        ),   
    dict(type='CutOut', 
    n_holes=7,
    cutout_shape=[(4, 4), (4, 8), (8, 4),
                       (8, 8), (16, 8), (8, 16),
                       (16, 16), (16, 32), (32, 16) ]),
    dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3,
                 bbox_clip_border=True),
    # dict(type='RandomAffine'),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


val_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(1024, 1024),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=[(1024, 1024),(512,512),(1333,800)],
            flip= False,
            flip_direction =  ["horizontal", "vertical" ,"diagonal"],
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]


data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=json_root + 'train_1.json',
        img_prefix=data_root ,
        classes = classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=json_root + 'val_1.json',
        img_prefix=data_root ,
        classes = classes,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root ,
        classes = classes,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox',    
                classwise=True,
                iou_thrs=[0.50],
                metric_items=['mAP','mAP_s','mAP_m','mAP_l'])

# https://github.com/open-mmlab/mmsegmentation/issues/122
