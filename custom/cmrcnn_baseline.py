_base_ = [
    '../configs/_base_/models/cascade-mask-rcnn_r50_fpn.py',
    'coco.py',
    'misc.py'
]

custom_imports = dict(
    imports=['custom.freeze_hook'], allow_failed_imports=False
)
custom_hooks = [dict(type='FreezeHook')]
