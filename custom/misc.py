_base_ = [
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]

# Optimizer settings
optimizer = dict(
    _delete_=True, 
    type='AdamW',  # Change to AdamW optimizer
    lr=0.0001, 
    betas=(0.9, 0.999), 
    weight_decay=0.05,
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.75)  # Add layer-wise learning rate decay
)

# Learning rate schedule
lr_config = dict(step=[27, 33])  # Update learning rate step schedule

# Runner settings
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)  # Use EpochBasedRunnerAmp with mixed precision

# Disable MMDet's built-in fp16
fp16 = None  # Explicitly disabling fp16

optimizer_config = dict(
    type="DistOptimizerHook", 
    update_interval=1, 
    grad_clip=None, 
    coalesce=True, 
    bucket_size_mb=-1, 
    use_fp16=True  # Use mixed-precision training with FP16
)

vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]
visualizer = dict(vis_backends=vis_backends)
visualization = dict(draw=True, show=True)

# MMEngine support the following two ways, users can choose
# according to convenience
default_hooks = dict(checkpoint=dict(interval=4))
train_cfg = dict(val_interval=2)
find_unused_parameters = True
