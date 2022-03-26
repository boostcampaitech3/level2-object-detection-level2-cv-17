checkpoint_config = dict(interval=1, max_keep_ckpts = 10)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='NumClassCheckHook'),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(
                project='mmdetection',
                entity = 'hyoseok',
                # group = None,
                # config = None,
                # job_type = f'Fold{fold}',
                reinit = True
            ),
            # with_step = False
            )
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook'),
                    # dict(type='MyHook',priority = 'LOWEST')
                    ]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
