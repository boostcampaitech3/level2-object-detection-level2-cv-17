# mmdet/datasets/builder.py line 76 cp_cfg.pop('type')
# 하단에 cp_cfg.pop('ann_file') 추가해줘야함.
# 무슨 문제인지.. mmdetection에 issue올려바도 될 듯

# TypeError: Albu: __init__() got an unexpected keyword argument 'img_scale'
# 이런 에러 뜨는 이유가, 뭔가 overwrite되서인 듯.
# ㄴㄴ 아님 내가 계속 arg[parse로] 줄 떄 index번호인 2 이런식으로주는데
# 실제 config에서는 안그래서 발생.

# default setting is fold 3
# if you want another fold modify traindataset

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

# (CutOut, MinIoURandomCrop, MixUp, Mosaic,
#   RandomAffine, RandomShift )

# # set train_pipeline


# nontype iterable 문제 발생 
# worker process 문제도 발생 -> 문제는 데이터 전달 시 오류
# augmentation logic상에 있을 거임.
# 나는 mosaic적용 후에, albuumentation을 적용해서 발생
# 그렇다면 albumentation을 적용한 거를 mosiac해야할까>?
# 아니면 애초에 mosaic할 때는 augmentation을 안해야할까

train_pipeline = [
    # 이 위 두줄이 엄청난 cost를 발생시켰던 것 같ㅊ음.
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True), # Bring Annotation boxes
    dict(type='Mosaic', img_scale=(512, 512),
                 center_ratio_range=(0.5, 1.5),
                 min_bbox_size=0.0,
                 bbox_clip_border=True,
                 skip_filter=True,
                 pad_val=114.0,
                 prob = 0.5),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-512 // 2, -512 // 2)),
    # dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(
    #     type='Albu',
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type='BboxParams',
    #         format='pascal_voc',
    #         label_fields=['gt_labels'],
    #         min_visibility=0.0,
    #         filter_lost_elements=True),
    #     keymap={
    #         'img': 'image',
    #         'gt_bboxes': 'bboxes'
    #     },
    #     update_pad_shape=False,
    #     skip_img_without_anno=True
    #     ),
    
    # dict(type='CutOut',  'n_holes': (10,20), 'cutout_shape': [(4,4), (4,8), (8,4)),
    # dict(type='MinIoURandomCrop', **img_norm_cfg),
    # dict(type='MixUp'), ? 성능이 무척 별로임.
    # dict(type='RandomAffine'),
    # dict(type='CutOut', 
    # n_holes=7,
    # cutout_shape=[(4, 4), (4, 8), (8, 4),
    #                    (8, 8), (16, 8), (8, 16),
    #                    (16, 16), (16, 32), (32, 16) ]),
    # dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
    #              min_crop_size=0.3,
    #              bbox_clip_border=True),

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

# Mosaic augmentation
# https://mmdetection--7507.org.readthedocs.build/en/7507/tutorials/how_to.html

train_dataset = dict(
    # _delete_ = True, # remove unnecessary Settings
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=json_root + 'train_3.json', 
        img_prefix=data_root,
        classes = classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        # filter_empty_gt=False,
    ),
    pipeline=train_pipeline
    )

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=json_root + 'val_3.json', # 이걸 쓸 떄는 cfg에서 다르게 fold를 줘야하네..
        # data.train.val.ann_file은 이걸로 즉, 이전에는 val_3이었음.
        # 다만, 학습할 때 train_1로 된 긋 train_3이 아니라.
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

