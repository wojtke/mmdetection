_base_ = [
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]

optim_wrapper = dict(
    _delete_=True
    type='AmpOptimWrapper',  
    optimizer=dict(
        type='AdamW',  
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        #paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.75) # TODO 
    ),
    paramwise_cfg=dict(
        num_layers=12,  # Set according to your model's depth
        layer_decay_rate=0.75  # The decay rate to apply per layer
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=None, 
    use_fp16=True
)

auto_scale_lr = dict(enable=True, base_batch_size=16)

train_cfg = dict(val_interval=1)


default_hooks = dict(
    visualization=dict(type='DetVisualizationHook', draw=True),
    checkpoint=dict(interval=1))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',)
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

find_unused_parameters = True