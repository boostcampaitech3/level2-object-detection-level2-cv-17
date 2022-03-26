# optimizer
optimizer = dict(type='Adam',
                lr=0.001,
                betas=(0.9, 0.999),
                weight_decay=0.05,
                # paramwise_cfg=dict(
                # custom_keys={
                # 'absolute_pos_embed': dict(decay_mult=0.),
                # 'relative_position_bias_table': dict(decay_mult=0.),
                # 'norm': dict(decay_mult=0.))
                )

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
        policy='CosineAnnealing',
        warmup='linear',
        warmup_iters=1000, 
        warmup_ratio=0.001,
        min_lr=1e-07,
    )
runner = dict(type='EpochBasedRunner', max_epochs=20)
